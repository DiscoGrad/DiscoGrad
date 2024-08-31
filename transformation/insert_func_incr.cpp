/** Contains an ASTVisitor that inserts the backend calls required
 *  for smoothing.
 *
 *  Copyright 2023, 2024 Philipp Andelfinger, Justin Kreikemeyer
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

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Rewrite/Frontend/Rewriters.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include <iostream>
#include <string>
#include <unordered_map>
#include "serialize.hpp"

using namespace clang;
using namespace clang::tooling;
using namespace llvm;
using namespace std;

const bool print_debug = false;

unordered_map<string, vector<int>> funcTransitiveSmoothBranches;

static cl::OptionCategory ToolCategory("Smooth");
Rewriter rewriter;

#define GET_LOC_BEFORE_END(S)                                                  \
  Lexer::getLocForEndOfToken(S->getEndLoc(), 0, srcMgr, langOpts)
#define GET_LOC_END(S) GET_LOC_BEFORE_END(S).getLocWithOffset(1)

/** Transforms the AST nodes of C++ functions to be smoothed 
 *  by inserting calls to the DGO backend 
 */
class SmoothVisitor : public RecursiveASTVisitor<SmoothVisitor> {
private:
  set<const Stmt *> LoopBodies;
  set<const Stmt *> SmoothIfElseBodies;
  set<const Stmt *> SmoothLoopBodies;
  set<const Stmt *> SmoothFuncBodies;

  string currFuncName;
  SourceLocation currSmoothFunctionEndLoc;
  bool currSmoothFunctionIsVoid = false;

  SourceManager &srcMgr;
  const LangOptions &langOpts;

  uint64_t nextBranchPos = 0;

public:
  explicit SmoothVisitor(ASTContext *Context) : Context(Context), srcMgr(Context->getSourceManager()), langOpts(Context->getLangOpts()) {
    rewriter.setSourceMgr(srcMgr, langOpts);
  }

  void collectCalledFuncs(Stmt *stmt, vector<string>& funcNames) {
    if (stmt == nullptr)
      return;

    if (isa<CXXMemberCallExpr>(stmt)) {
      CXXMemberCallExpr *e = cast<CXXMemberCallExpr>(stmt);
      if (isa<CXXMethodDecl>(e->getCalleeDecl())) {
        CXXMethodDecl *callee = cast<CXXMethodDecl>(e->getCalleeDecl());
        string funcName = callee->getNameInfo().getAsString();
        funcNames.push_back(funcName);
      }
    }
    for (auto it = stmt->children().begin(); it != stmt->children().end(); it++)
      collectCalledFuncs(*it, funcNames);
  }

  bool needsSmoothing(Stmt *stmt) {
    if (isa<Expr>(stmt)) {
      Expr *e = cast<Expr>(stmt);
      string typeStr = e->getType().getAsString();
      if (typeStr.find("adouble") != string::npos) {
        return true;
      }
    }
    for (auto it = stmt->children().begin(); it != stmt->children().end(); it++) {
      if (needsSmoothing(*it))
        return true;
    }
    return false;
  }

  bool VisitStmt(Stmt *s) {

    if (print_debug) {
      string out_str;
      raw_string_ostream outstream(out_str);
      s->printPretty(outstream, NULL, PrintingPolicy(langOpts));
      cerr << "current statement: " << out_str << endl;
    }

    // to force initialization of the rewrite buffer
    rewriter.InsertText(s->getBeginLoc(), "");

    if (s->getBeginLoc() >= currSmoothFunctionEndLoc) {
      return true;
    }

    if (isa<IfStmt>(s)) {
      IfStmt *If = cast<IfStmt>(s);
      Stmt *Else = If->getElse();
      Expr *Cond = If->getCond();

      bool smoothedBranch = false;
      if (needsSmoothing(Cond)) {

        auto CondText = rewriter.getRewrittenText(Cond->getSourceRange());

        string unhandledOps[] = { "||", "&&", "==", "!=" };

        bool skip = false;
        for (auto &op : unhandledOps)
          if (CondText.find(op) != string::npos)
            skip = true;

        if (!skip) {
          string leCmps[] = { "<=", "<" };
          for (const string& cmpOp : leCmps) {
            auto cmpPos = CondText.find(cmpOp);
            if (cmpPos != string::npos) {
              CondText.replace(cmpPos, cmpOp.length(), "-(");
              CondText += ")";
              smoothedBranch = true;
            }
          }

          auto geCmps = { ">=", ">" };
          for (const string& cmpOp : geCmps) {
            auto cmpPos = CondText.find(cmpOp);
            if (cmpPos != string::npos) {
              CondText = CondText.substr(cmpPos + cmpOp.length()) + "- (" + CondText.substr(0, cmpPos) + ")";
              smoothedBranch = true;
            }
          }

          //if (smoothedBranch) {
          //  auto condVarName = "_discograd_cond_" + to_string(nextBranchPos);
          //  rewriter.InsertText(If->getBeginLoc(), "\nadouble " + condVarName + " = " + CondText + ";\n"); 
          //  auto endBlockLoc = GET_LOC_BEFORE_END(Else);
          //  rewriter.InsertText(If->getBeginLoc(), "\n_discograd.prepare_branch(" + to_string(nextBranchPos) + ", " + condVarName + ");\n"); 
          //  rewriter.InsertText(Cond->getBeginLoc(), condVarName + " < 0.0 /*"); 
          //  rewriter.InsertText(GET_LOC_BEFORE_END(Cond), " */"); 

          //  rewriter.InsertText(endBlockLoc, "\n_discograd.end_block();\n");
          //}
        }
      }

      Stmt *Then = If->getThen();
      //vector<uint64_t> thenBranchPositions;
      //vector<uint64_t> elseBranchPositions;
      //uint64_t thenIfs = countNestedIfs(Then);
      //uint64_t elseIfs = countNestedIfs(Else);

      //uint64_t bpos = nextBranchPos + smoothedBranch;
      //for (int i = 0; i < thenIfs; i++) {
      //  rewriter.InsertText(Else->getBeginLoc().getLocWithOffset(1), "\n_discograd.inc_branch_visit(" + to_string(bpos) + ");\n");
      //  bpos++;
      //}
      //for (int i = 0; i < elseIfs; i++) {
      //  rewriter.InsertText(Then->getBeginLoc().getLocWithOffset(1), "\n_discograd.inc_branch_visit(" + to_string(bpos) + ");\n");
      //  bpos++;
      //}

      vector<string> thenCalledFuncs, elseCalledFuncs;
      collectCalledFuncs(Then, thenCalledFuncs);
      collectCalledFuncs(Else, elseCalledFuncs);

      for (auto& f : thenCalledFuncs)
        for (auto bpos : funcTransitiveSmoothBranches[f])
          rewriter.InsertText(Else->getBeginLoc().getLocWithOffset(1), "\n_discograd.inc_branch_visit(" + to_string(bpos) + ");\n");

      for (auto& f : elseCalledFuncs)
        for (auto bpos : funcTransitiveSmoothBranches[f])
          rewriter.InsertText(Then->getBeginLoc().getLocWithOffset(1), "\n_discograd.inc_branch_visit(" + to_string(bpos) + ");\n");

      auto endBlockLoc = GET_LOC_BEFORE_END(Else);

      if (smoothedBranch) {
        nextBranchPos++;
      }
      
    }
    

    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *f) {
    if (f->hasBody()) {
      SourceManager &srcMgr = rewriter.getSourceMgr();
      const LangOptions &langOpts = rewriter.getLangOpts();

      Stmt *FuncBody = f->getBody();

      QualType QT = f->getReturnType();
      string TypeStr = QT.getAsString();

      DeclarationName DeclName = f->getNameInfo().getName();
      string FuncName = DeclName.getAsString();

      size_t found = FuncName.find("_DiscoGrad_");
      if (found != string::npos) {
        currSmoothFunctionIsVoid = !TypeStr.compare("void");

        currSmoothFunctionEndLoc = GET_LOC_END(FuncBody);
        
        currFuncName = FuncName;

        SmoothFuncBodies.insert(FuncBody);
     }
   }

    return true;
  }

private:
  ASTContext *Context;
};

class SmoothConsumer : public clang::ASTConsumer {
public:
  explicit SmoothConsumer(ASTContext *Context) : Visitor(Context) {}

  virtual void HandleTranslationUnit(clang::ASTContext &Context) {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  SmoothVisitor Visitor;

};

class SmoothAction : public clang::ASTFrontendAction {
public:
  virtual unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler, StringRef InFile) {
    return make_unique<SmoothConsumer>(&Compiler.getASTContext());
  }
};

int main(int argc, const char **argv) {
  Expected<CommonOptionsParser> op =
      CommonOptionsParser::create(argc, argv, ToolCategory);
  if (auto err = op.takeError()) {
    logAllUnhandledErrors(std::move(err), errs(), "Error ");
    return -1;
  }
  ClangTool Tool(op->getCompilations(), op->getSourcePathList());

  string inFname = op->getSourcePathList()[0];
  string smoothBranchesFname = inFname.substr(0, inFname.length() - strlen("smoothed.cpp")) + "smoothBranches.bin";

  funcTransitiveSmoothBranches = deserialize(smoothBranchesFname);

  int r = Tool.run(newFrontendActionFactory<SmoothAction>().get());

  const RewriteBuffer *RewriteBuf =
      rewriter.getRewriteBufferFor(rewriter.getSourceMgr().getMainFileID());
  assert(RewriteBuf);
  cout << string(RewriteBuf->begin(), RewriteBuf->end());

  return r;
}
