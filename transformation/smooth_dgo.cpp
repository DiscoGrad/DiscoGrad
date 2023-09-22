/** Contains an ASTVisitor that inserts the proper backend calls
 *  for smoothing.
 *
 *  Copyright 2023 Philipp Andelfinger, Justin Kreikemeyer
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

using namespace clang;
using namespace clang::tooling;
using namespace llvm;
using namespace std;

const bool print_debug = false;

static cl::OptionCategory ToolCategory("Smooth");
Rewriter rewriter;

#define GET_LOC_BEFORE_END(S)                                                  \
  Lexer::getLocForEndOfToken(S->getEndLoc(), 0, srcMgr, langOpts)
#define GET_LOC_END(S) GET_LOC_BEFORE_END(S).getLocWithOffset(1)

/** Transforms the AST nodes of C++ functions to be smoothed 
 *  into smoothed versions by inserting the proper backend 
 *  (cf. {@link si.hpp}) calls.
 */
class SmoothVisitor : public RecursiveASTVisitor<SmoothVisitor> {
private:
  set<const Stmt *> LoopBodies;
  set<const Stmt *> SmoothIfElseBodies;
  set<const Stmt *> SmoothLoopBodies;
  set<const Stmt *> SmoothFuncBodies;

  SourceLocation currSmoothFunctionEndLoc;
  bool currSmoothFunctionIsVoid = false;

  SourceManager &srcMgr;
  const LangOptions &langOpts;

public:
  explicit SmoothVisitor(ASTContext *Context) : Context(Context), srcMgr(Context->getSourceManager()), langOpts(Context->getLangOpts()) {
    rewriter.setSourceMgr(srcMgr, langOpts);
  }

  bool needsSmoothing(Stmt *stmt) {
    if (isa<Expr>(stmt)) {
      Expr *e = cast<Expr>(stmt);
      string typeStr = e->getType().getAsString();
      if (typeStr.find("sdouble") != string::npos)
        return true;
    }
    for (auto it = stmt->children().begin(); it != stmt->children().end(); it++) {
      if (needsSmoothing(*it))
        return true;
    }
    return false;
  }

  bool crispUpToOutermostLoop(Stmt *stmt) {
    DynTypedNodeList parents = Context->getParents(*stmt);
    while (!parents.empty()) {
      const Stmt *p = parents[0].get<Stmt>();
      if (!p)
        break;
      if (SmoothIfElseBodies.contains(p))
        return false;
      if (LoopBodies.contains(p))
        return !SmoothLoopBodies.contains(p);

      parents = Context->getParents(*p);
    }
    assert(false);
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
      Expr *Cond = If->getCond();

      if (!needsSmoothing(Cond))
        return true;

      auto CondText = rewriter.getRewrittenText(Cond->getSourceRange());

      // FIXME: problem with the following:
      // if (x < dist(x)), where dist(x) is non-deterministic results in the wrong expression
      // one idea is to store dist(x) in a temp value, but would need to do this for
      // all function calls inside Cond.

      //cerr << "CondText is " << CondText << endl;
      if (CondText.find("||") != string::npos || CondText.find("&&") != string::npos)
        return true;

      // cant smooth (in)equality for now
      if (CondText.find("==") != string::npos || CondText.find("!=") != string::npos)
        return true;

      bool foundComparison = false;

      // replace all <= with -
      if (CondText.find("<=") != string::npos) {
        CondText.replace(CondText.find("<="), 2, "-(");
        foundComparison = true;
      }
      // replace all >= with *(-1)+
      if (CondText.find(">=") != string::npos) {
        CondText.replace(CondText.find(">="), 2, "*(-1)+(");
        foundComparison = true;
      }
      // replace all < with -
      if (CondText.find("<") != string::npos) {
        CondText.replace(CondText.find("<"), 1, "-(");
        foundComparison = true;
      }
      // replace all > with *(-1)+
      if (CondText.find(">") != string::npos) {
        CondText.replace(CondText.find(">"), 1, "*(-1)+(");
        foundComparison = true;
      }


      // if there is still no comparison operator, then we handle it like equality
      if (!foundComparison) {
        return true;
      }

      rewriter.InsertText(
        If->getBeginLoc(),
        "\n_discograd.prepare_branch(" + CondText + "));\n"
      ); 
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
        assert(!TypeStr.compare("void") || !TypeStr.compare("sdouble") || !TypeStr.compare("class sdouble")  ||
                                           !TypeStr.compare("adouble") || !TypeStr.compare("class adouble"));

        currSmoothFunctionEndLoc = GET_LOC_END(FuncBody);

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

  int r = Tool.run(newFrontendActionFactory<SmoothAction>().get());

  const RewriteBuffer *RewriteBuf =
      rewriter.getRewriteBufferFor(rewriter.getSourceMgr().getMainFileID());
  assert(RewriteBuf);
  cout << string(RewriteBuf->begin(), RewriteBuf->end());

  return r;
}
