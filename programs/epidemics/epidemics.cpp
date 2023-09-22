/*  Copyright 2023 Philipp Andelfinger, Justin Kreikemeyer
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 *  and associated documentation files (the “Software”), to deal in the Software without
 *  restriction, including without limitation the rights to use, copy, modify, merge, publish,
 *  distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
 *  Software is furnished to do so, subject to the following conditions:
 *   
 *    The above copyright notice and this permission notice shall be included in all copies or
 *    substantial portions of the Software.
 *    
 *    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 *    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 *    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 *    ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 *    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *    SOFTWARE.
 */

/* 
  Parameters:
    - mean recovery time (= mean of exponentially distributed sojourn time) relative to end time
    - initial infection probability (equal for all agents)
    - for each location: per-location infection probability

 * Experiment Preparation
 * ======================
 *
 * - use `generate_args.py <num_inputs>` and `generate_network.py <num_locations>` in this folder
 * - use `cat /programs/epidemics/args.txt|./programs/epidemics/epidemics_crisp ...` or similar
 *   to generate a reference trajectory
 * - view the generated reference with `plot_reference.py` and repeat the above steps until satisfied with the output
 */

#include <stdio.h>

#include <stdlib.h>
#include <array>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <unistd.h>

using namespace std;

string exec_path;

const int nLocs = 100;
const int nAgents = 200;
const int endTime = 25;

const int num_inputs = nLocs + 2;

#include "backend/discograd.hpp"

//#define DEBUG
#ifdef DEBUG
#define printf_debug(str, args...) printf(str, args);
#define print_debug(str) printf(str);
#else
#define printf_debug(str, args...) 
#define print_debug(str) 
#endif

// simulation constants: susceptible, infected, recovered
static const int num_states = 3; 
double SUS = 0.0;
double INF = 1.0;
double REC = 2.0;

// simulation trace
struct Hist {
  double s[num_states]; 
};

// randomness
const int loc_seed = 1234567;      // the same for each replication
default_random_engine loc_gen;     // for easy reproducible gen of locations
sdouble uniform_to_exp(double u, sdouble& mean)
{
    return -mean * log(u);
}

// network environment
vector<vector<int>> network;
void load_network(int nLocs) // to be generated with generate_network.py
{
  string network_file = exec_path + "/network.txt";
  if(network.size() > 0) // already loaded
    return;

  network.resize(nLocs);

  const int buf_len = 1024;
  char buf[buf_len];
  FILE *fp = fopen(network_file.c_str(), "r");
  if(fp == NULL) {
    printf("cannot open network file %s for reading\n", network_file.c_str());
    exit(1);
  }

  int n0, n1;
  while(fgets(buf, buf_len, fp)) {
    sscanf(buf, "%d %d", &n0, &n1);
    network[n0].push_back(n1);
    network[n1].push_back(n0);
  }
  fclose(fp);
}

// trajectory input/output
void load_states(Hist out_states[][nLocs])
{
  string filename = exec_path + "/reference.csv";
  const int buf_len = 1024;
  char buf[buf_len];
  FILE *fp = fopen(filename.c_str(), "r");
  if(fp == NULL) {
    printf("cannot open states file %s for reading, will create it.\n", filename.c_str());
    return;
  }

  double sus, inf, rec;
  for (int t = 0; t < endTime; ++t)  {
    for (int loc = 0; loc < nLocs; loc++) {
      fgets(buf, buf_len, fp);
      sscanf(buf, "%lf, %lf, %lf", &sus, &inf, &rec);
      out_states[t][loc].s[0] = sus;
      out_states[t][loc].s[1] = inf;
      out_states[t][loc].s[2] = rec;
    }
  }
  fclose(fp);
}

void write_states(Hist states[][nLocs])
{
  string filename = exec_path + "/reference.csv";

  if (access(filename.c_str(), F_OK) == 0) // file exists
    return;

  FILE *fp = fopen(filename.c_str(), "w");
  if (fp == NULL) {
    printf("cannot open trajectory file %s for writing.\n", filename.c_str());
    exit(1);
  }

  for (int t = 0; t < endTime; ++t)  {
    for (int i = 0; i < nLocs; ++i)
    {
      fprintf(fp, "%lf, %lf, %lf\n", states[t][i].s[0], states[t][i].s[1], states[t][i].s[2]);
    }
  }
  fclose(fp);
}


// trajectory input/output
//void load_trajectory(int nStates, Hist out_trajectory[])
//{
//  string filename = exec_path + "/reference.csv";
//  const int buf_len = 1024;
//  char buf[buf_len];
//  FILE *fp = fopen(filename.c_str(), "r");
//  if(fp == NULL) {
//    printf("cannot open trajectory file %s for reading.\n", filename.c_str());
//    exit(1);
//  }
//
//  double sus, inf, rec;
//  int i = 0;
//  while(fgets(buf, buf_len, fp)) {
//    sscanf(buf, "%lf, %lf, %lf", &sus, &inf, &rec);
//    out_trajectory[i].s[0] = sus;
//    out_trajectory[i].s[1] = inf;
//    out_trajectory[i].s[2] = rec;
//    i++;
//  }
//  fclose(fp);
//}
//void write_trajectory(Hist trajectory[], int nStates)
//{
//  string filename = exec_path + "/reference.csv";
//  FILE *fp = fopen(filename.c_str(), "w");
//  if (fp == NULL) {
//    printf("cannot open trajectory file %s for writing.\n", filename.c_str());
//    exit(1);
//  }
//
//  for (int i = 0; i < nStates; ++i)
//  {
//    fprintf(fp, "%lf, %lf, %lf\n", trajectory[i].s[0], trajectory[i].s[1], trajectory[i].s[2]);
//  }
//  fclose(fp);
//}

void print_state(const vector<sdouble>& s, const vector<vector<int>>& loc_to_agents, const vector<sdouble>& loc_infection_prob)
{
  int loc_c = 0;
  for (auto loc_it = loc_to_agents.begin(); loc_it != loc_to_agents.end(); ++loc_it)
  {
    auto loc = *loc_it;
    printf("loc %d (%lf%%): ", loc_c, loc_infection_prob[loc_c].expectation().val*100.0);
    for (auto a_it = loc.begin(); a_it != loc.end(); ++a_it)
    {
      int a = *a_it;
      printf("a%d(%s), ", a, s[a].expectation().val == 0 ? "sus" : s[a].expectation().val  == 1 ? "inf" : "rec");
    }
    printf("\n");
    loc_c++;
  }
}

/** An agent-based SIR simulation */
adouble _DiscoGrad_epidemics(DiscoGrad<num_inputs> &_discograd, array<adouble, num_inputs> &x, Hist ref_states[][nLocs], Hist out_states[][nLocs], int nAgents, int endTime)
{
  // model inputs
  sdouble mean_recovery_time({x[0], _discograd.get_variance()}); // mean recovery time (= mean of exponentially distributed sojourn time) relative to end time
  mean_recovery_time *= endTime;
  sdouble init_infected_prob({x[1], _discograd.get_variance()});  // initial infection probability (equal for all agents)
  vector<sdouble> loc_infection_prob(nLocs);      // per-location infection probability
  for (int loc = 0; loc < nLocs; ++loc)
    loc_infection_prob[loc] = sdouble({x[loc+2], _discograd.get_variance()});

  // list of agents (double-buffered)
  sdouble recovery_timer[nAgents];                       // recovery timer for each agent
  vector<sdouble> s(nAgents), s_buff(nAgents);           // SIR state for each agent (smooth)
  vector<vector<int>> loc_to_agents, loc_to_agents_buff; // mapping from location id to agent id
  loc_to_agents.resize(nLocs);
  loc_to_agents_buff.resize(nLocs);
  vector<int> agent_to_loc, agent_to_loc_buff;           // mapping from agent id to location id
  agent_to_loc.resize(nAgents);
  agent_to_loc_buff.resize(nAgents);

  // initialize agents
  loc_gen.seed(loc_seed);
  uniform_real_distribution<double> uniform_dist(0.0, 1.0);
  for (int a = 0; a < nAgents; ++a)
  {
    s[a] = s_buff[a] = 0.0;
    // set location
    int loc = (int)(uniform_dist(loc_gen) * nLocs);
    agent_to_loc[a] = agent_to_loc_buff[a] = loc;
    loc_to_agents[loc].push_back(a);
    loc_to_agents_buff[loc].push_back(a);
    // set initial infection state and recovery time
    sdouble recovery_time = 0.0;
    sdouble state = 0;
    double infect = uniform_dist(_discograd.rng);
    sdouble temp_rec = uniform_to_exp(uniform_dist(_discograd.rng), mean_recovery_time);
    if (init_infected_prob > infect)
    {
      state = INF;
      recovery_time = temp_rec;
    }
    s[a] = s_buff[a] = state;
    recovery_timer[a] = recovery_time;
  }
#ifdef DEBUG
  printf("initial state:\n");
  print_state(s, loc_to_agents, loc_infection_prob);
#endif

  // the final output
  sdouble loss = 0.0;

  for (int t = 0; t < endTime; ++t) 
  {
#ifdef DEBUG
    double sim_time = t;
#endif
    for (int a = 0; a < nAgents; ++a)
    {
      int loc = agent_to_loc[a];
      // if infected, progress recovery
      recovery_timer[a] -= 1.0;
      if (recovery_timer[a] <= 0.0)
      {
        if (s[a] == INF)
        {
          s_buff[a] = REC;
        }
      }
      // else, if susceptible to the disease, (possibly) get infected by neighbours
      sdouble infection_prob = loc_infection_prob[loc];
      sdouble infect = uniform_dist(_discograd.rng);
      sdouble temp_recovery = uniform_to_exp(uniform_dist(_discograd.rng), mean_recovery_time);
      if (infection_prob > infect) 
      {
        vector<int> neighbours = loc_to_agents[loc];
        for (auto n_it = neighbours.begin(); n_it != neighbours.end(); ++n_it)
        { 
          int n = *n_it;
          if (a != n)
          {
            //if (n == a) continue; // agent cannot be its own neighbour
            //if (s[n] == INF) { printf_debug("agent a%d has neighbour a%d(inf)\n", a, n); }
            if (s[a] == SUS)
            {
              if (s[n] == INF)
              {
                s_buff[a] = INF;
                recovery_timer[a] = temp_recovery;
                //printf_debug("Agent a%d is now infected. Recovery scheduled at %lf\n", a, recovery_timer[a].expectation().val);
                //break;
              }
            }
          }
        }
      }

      // move agent to new random location
      vector<int>& adj_locs = network[loc];
      if (!adj_locs.empty())
      {
        int next_loc = adj_locs[(int)(uniform_dist(loc_gen) * adj_locs.size())];
        agent_to_loc_buff[a] = next_loc;
        printf_debug("agent %d moves from %d to %d\n", a, loc, next_loc);
        loc_to_agents_buff[loc].erase(
          remove(
            loc_to_agents_buff[loc].begin(), 
            loc_to_agents_buff[loc].end(),
            a
          ), 
          loc_to_agents_buff[loc].end()
        );
        loc_to_agents_buff[next_loc].push_back(a);
      }
    }

    // swap double buffers
    for (int a = 0; a < nAgents; ++a)
    {
      s[a] = s_buff[a];
      agent_to_loc[a] = agent_to_loc_buff[a];
    }
    for (int loc = 0; loc < nLocs; ++loc)
    {
      loc_to_agents[loc] = loc_to_agents_buff[loc];
    }

    sdouble out_hist[nLocs][num_states];

    // build histogram and update loss
    for (int a = 0; a < nAgents; ++a)
    {
      int loc = agent_to_loc[a];

      if (s[a] == SUS) out_hist[loc][0] += 1;
      else if (s[a] == INF) out_hist[loc][1] += 1;
      else out_hist[loc][2] += 1;
    }
    printf_debug("%d, ", t);
    for (int loc = 0; loc < nLocs; loc++) {
      for (int i = 0; i < num_states; ++i) { 
        // update output states
        out_states[t][loc].s[i] = out_hist[loc][i].expectation().val;
        printf_debug("%f, ", out_states[t][loc].s[i]);
        // update loss
        printf_debug("ref %lf out %lf\n", ref_states[loc].s[i], out_hist[loc][i].expectation().get_val());
        sdouble err = ref_states[t][loc].s[i] - out_hist[loc][i];
        loss += err * err / nLocs;
      }
    }
    print_debug("\n");

#ifdef DEBUG
  printf("state after time %d(%lf):\n", t, sim_time);
  print_state(s, loc_to_agents, loc_infection_prob);
#endif

  }

  return loss.expectation();
}


class Epidemics : public DiscoGradProgram<num_inputs> {
private:
  // default simulation (hyper-)parameters
  int nRuns = 0;
  Hist ref_states[endTime][nLocs];
  Hist out_states[endTime][nLocs];
public:
  Epidemics() {
    // load reference trajectory (= data for calibration) and prepare output trajectory
    printf_debug("nLocs: %d\n", nLocs);
    for (int t = 0; t < endTime; t++) {
      for (int i = 0; i < nLocs; ++i) {
        for (int j = 0; j < num_states; ++j) {
          out_states[t][i].s[j] = 0.0;
          ref_states[t][i].s[j] = 0.0;
        }
      }
    }
    load_states(ref_states);

    // load network environment
    load_network(nLocs);
  }
  adouble run(DiscoGrad<num_inputs> &_discograd, array<adouble, num_inputs> &p) {
    nRuns++;
    // execute (smoothed) simulation and update states
    Hist buff_states[endTime][nLocs];
    for (int t = 0; t < endTime; t++)
      for (int i = 0; i < nLocs; ++i)
        for (int j = 0; j < num_states; ++j)
          buff_states[t][i].s[j] = 0.0;
    adouble y = _DiscoGrad_epidemics(_discograd, p, ref_states, buff_states, nAgents, endTime);

    // write output states
    for (int t = 0; t < endTime; t++)
      for (int i = 0; i < nLocs; ++i)
        for (int j = 0; j < num_states; ++j)
          out_states[t][i].s[j] += buff_states[t][i].s[j];
     
    // debugging 
    printf_debug("rexpectation: %lf\n", y.val);
    for (int i = 0; i < num_inputs; ++i) {
      printf_debug("rderivative: %lf\n", y.get_adj(i));
    }

    return y;
  }

  void write_output() {
    for (int t = 0; t < endTime; t++) {
      for (int i = 0; i < nLocs; ++i) {
        for (int j = 0; j < num_states; ++j) {
          out_states[t][i].s[j] /= nRuns;
          printf("%lf, ", out_states[t][i].s[j]);
        }
        printf("\n");
      }
    }
    write_states(out_states);
  }
};

int main(int argc, char **argv)
{
  exec_path = filesystem::path(argv[0]).parent_path().generic_string();

  DiscoGrad<num_inputs> dg(argc, argv);
  Epidemics prog;
  dg.estimate(prog);
  prog.write_output();
  return 0;
}
