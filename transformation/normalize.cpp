/** Contains an ASTVisitor that inserts otherwise optional brackets
 *  for all scoped code, for example in short if statements, to prepare
 *  the code for the smoothing ASTVisitor.
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
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Rewrite/Frontend/Rewriters.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include <iostream>

using namespace clang;
using namespace clang::tooling;
using namespace llvm;
using namespace std;

const bool print_debug = false;

static cl::OptionCategory ToolCategory("Normalize");
Rewriter rewriter;

#define GET_LOC_BEFORE_END(S)                                                  \
  Lexer::getLocForEndOfToken(S->getEndLoc(), 0, srcMgr, langOpts)
#define GET_LOC_END(S) GET_LOC_BEFORE_END(S).getLocWithOffset(1)

/** Transforms the AST nodes of the functions to be smoothed,
 *  such that each control flow statement has the (otherwise optional)
 *  brackets for scoping. This is necessary so that the smoothing visitor
 *  can safely insert backend calls.
 */
class NormalizeVisitor : public RecursiveASTVisitor<NormalizeVisitor> {
private:
  SourceLocation currSmoothFunctionEndLoc;
  SourceManager &srcMgr;
  const LangOptions &langOpts;

public:
  explicit NormalizeVisitor(ASTContext *Context) : Context(Context), srcMgr(Context->getSourceManager()), langOpts(Context->getLangOpts()) {
    rewriter.setSourceMgr(srcMgr, langOpts);
  }

  /** Insert additional {} around statements for scoping. */
  bool VisitStmt(Stmt *s) {
    SourceManager &srcMgr = rewriter.getSourceMgr();
    const LangOptions &langOpts = rewriter.getLangOpts();

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

      Stmt *Then = If->getThen();

      if (!isa<CompoundStmt>(Then)) {
        rewriter.InsertText(Then->getBeginLoc(), "{");
        rewriter.InsertText(GET_LOC_END(Then), "}");
      }

      Stmt *Else = If->getElse();
      if (Else && !isa<CompoundStmt>(Else)) {
          rewriter.InsertText(Else->getBeginLoc(), "{");
          rewriter.InsertText(GET_LOC_END(Else), "}");
      }
    }

    if (isa<ForStmt>(s)) {
      ForStmt *For = cast<ForStmt>(s);

      // move initializer outside the statement, in a new scope
      Stmt *Init = For->getInit();
      auto InitText = rewriter.getRewrittenText(Init->getSourceRange());

      rewriter.InsertText(For->getBeginLoc(), "{\n" + InitText + "\n");
      rewriter.InsertText(GET_LOC_END(For), "\n}\n");

      rewriter.InsertText(Init->getBeginLoc(), "/* ");
      rewriter.InsertText(GET_LOC_BEFORE_END(Init).getLocWithOffset(-1), " */");

      Stmt *ForBody = For->getBody();
      if (!isa<CompoundStmt>(ForBody)) {
        rewriter.InsertText(ForBody->getBeginLoc(), "{");
        rewriter.InsertText(GET_LOC_END(ForBody), "}");
      }
    }

    if (isa<WhileStmt>(s)) {
      WhileStmt *While = cast<WhileStmt>(s);

      Stmt *WhileBody = While->getBody();
      if (!isa<CompoundStmt>(WhileBody)) {
        rewriter.InsertText(WhileBody->getBeginLoc(), "{");
        rewriter.InsertText(GET_LOC_END(WhileBody), "}");
      }
    }

    if (isa<DoStmt>(s)) {
      DoStmt *Do = cast<DoStmt>(s);

      Stmt *DoBody = Do->getBody();
      if (!isa<CompoundStmt>(DoBody)) {
        rewriter.InsertText(DoBody->getBeginLoc(), "{");
        rewriter.InsertText(GET_LOC_END(DoBody), "}");
      }
    }

    return true;
  }

  /** Detect smooth functions (prefix _DiscoGrad_). */
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
        currSmoothFunctionEndLoc = GET_LOC_END(FuncBody);
      }

    }

    return true;
  }

private:
  ASTContext *Context;
};

class NormalizeConsumer : public clang::ASTConsumer {
public:
  explicit NormalizeConsumer(ASTContext *Context) : Visitor(Context) {}

  virtual void HandleTranslationUnit(clang::ASTContext &Context) {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  NormalizeVisitor Visitor;

};

class NormalizeAction : public clang::ASTFrontendAction {
public:
  virtual unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler, StringRef InFile) {
    return make_unique<NormalizeConsumer>(&Compiler.getASTContext());
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

  int r = Tool.run(newFrontendActionFactory<NormalizeAction>().get());

  const RewriteBuffer *RewriteBuf =
      rewriter.getRewriteBufferFor(rewriter.getSourceMgr().getMainFileID());

  assert(RewriteBuf);
  cout << string(RewriteBuf->begin(), RewriteBuf->end());

  return r;
}
