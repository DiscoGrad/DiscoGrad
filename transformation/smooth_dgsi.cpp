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

using namespace clang;
using namespace clang::tooling;
using namespace llvm;
using namespace std;

const bool print_debug = false;

#ifndef SMOOTH_LOOPS
#define SMOOTH_LOOPS false
#endif
const bool smoothLoops = SMOOTH_LOOPS; // experimental

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

      if (print_debug)
        cerr << "checking expression of type " << typeStr << endl;

      if (typeStr.find("sdouble") != string::npos) {
         if (print_debug)
           cerr << "found type " << typeStr << ", branch will be smoothed" << endl;

        return true;
      }
    }
    for (auto it = stmt->children().begin(); it != stmt->children().end(); it++) {
      if (needsSmoothing(*it))
        return true;
    }

    if (print_debug) {
      string out_str;
      raw_string_ostream outstream(out_str);
      stmt->printPretty(outstream, NULL, PrintingPolicy(langOpts));
      cerr << "statement " << out_str << " doesn't need smoothing" << endl;
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

    if (smoothLoops) {
      if (isa<BreakStmt>(s)) {
        if (crispUpToOutermostLoop(s)) {
          rewriter.InsertText(s->getBeginLoc(), "si_stack.exit_scope();\n");
        } else {
          rewriter.InsertText(s->getBeginLoc(), "si_stack.break_(); /* ");
          rewriter.InsertText(GET_LOC_END(s), " */");
        }
        return true;
      }

      if (isa<ContinueStmt>(s)) {
        if (crispUpToOutermostLoop(s)) {
          rewriter.InsertText(s->getBeginLoc(), "si_stack.exit_scope();\n");
        } else {
          rewriter.InsertText(s->getBeginLoc(), "si_stack.continue_(); /* ");
          rewriter.InsertText(GET_LOC_END(s), " */");
        }
        return true;
      }

      if (isa<ReturnStmt>(s)) {
        ReturnStmt *returnStmt = cast<ReturnStmt>(s);
        Expr *retVal = returnStmt->getRetValue();
        string retValText = "";
        if (retVal != NULL)
          retValText = rewriter.getRewrittenText(retVal->getSourceRange());

        rewriter.InsertText(returnStmt->getBeginLoc(), "si_return_val = " + retValText + ";\nsi_stack.return_(); /* ");
        rewriter.InsertText(GET_LOC_END(returnStmt), " */");

        return true;
      }
    }

    if (isa<DeclStmt>(s)) {
      DeclStmt *declStmt = cast<DeclStmt>(s);
      DeclGroupRef declGroup = declStmt->getDeclGroup();
      string outStr;
      bool skip = false;
      for (auto it = declGroup.begin(); it != declGroup.end(); it++) {
        Decl *decl = *it;
        if (isa<VarDecl>(decl)) {
          VarDecl *varDecl = cast<VarDecl>(decl);
          auto varDeclText =
              rewriter.getRewrittenText(varDecl->getSourceRange());
          string typeStr = varDecl->getType().getAsString();

          // TODO: skip reference initializations
          if (typeStr.back() == '&') {
            skip = true;
            break;
          }

          if (typeStr.find("sdouble") == string::npos) {
            skip = true;
            break;
          }

          // skip templated variables for now
          if (typeStr.back() == '>') {
            skip = true;
            break;
          }

          string nameStr = varDecl->getName().data();
          string fullNameStr = nameStr;
          size_t dim_start = typeStr.find("[");
          if (dim_start != string::npos) {
            fullNameStr += typeStr.substr(dim_start);
            typeStr = typeStr.substr(0, dim_start);
          }
          if (varDecl->hasInit()) {
            Expr *init = varDecl->getInit();
            auto initText = rewriter.getRewrittenText(init->getSourceRange());
            if (varDecl->getInitStyle() == 0) {
              outStr += typeStr + " " + fullNameStr + ";\n";
              outStr += nameStr + " = " + initText + ";\n";
            } else {
              if (initText == nameStr) {
                outStr += typeStr + " " + fullNameStr + ";\n";
              } else {
                outStr += typeStr + " " + initText + ";\n";
              }
            }
          }
        }
      }
      if (!skip) {
        rewriter.InsertText(s->getBeginLoc(), "/* ");
        rewriter.InsertText(GET_LOC_BEFORE_END(s), " */\n" + outStr);
      }
    }

    if (isa<CompoundStmt>(s) && !SmoothIfElseBodies.contains(s) &&
        !LoopBodies.contains(s) && !SmoothFuncBodies.contains(s)) {
      rewriter.InsertText(s->getBeginLoc().getLocWithOffset(1),
                          "\nsi_stack.enter_scope();");
      rewriter.InsertText(GET_LOC_BEFORE_END(s).getLocWithOffset(-1),
                          "si_stack.exit_scope();\n");
    }

    if (isa<IfStmt>(s)) {
      IfStmt *If = cast<IfStmt>(s);
      Expr *Cond = If->getCond();

      if (!needsSmoothing(Cond))
        return true;

      auto CondText = rewriter.getRewrittenText(Cond->getSourceRange());

      rewriter.InsertText(
          If->getBeginLoc(),
          "\nsi_stack.prepare_branch();\n"
          "\n{ SiPathWeights si_then_weights = (SiPathWeights)(" + CondText +
              ");");
      rewriter.InsertText(If->getBeginLoc(), "\n/* ");
      rewriter.InsertText(If->getRParenLoc().getLocWithOffset(1), " */");

      Stmt *Then = If->getThen();
      SmoothIfElseBodies.insert(Then);

      assert(isa<CompoundStmt>(Then));
      rewriter.InsertText(Then->getBeginLoc().getLocWithOffset(1),
                            "\nsi_stack.enter_if(si_then_weights);\n"
                            "if (!si_stack.top().empty()) {\n");

      rewriter.InsertText(GET_LOC_BEFORE_END(Then), "\n}\n");

      Stmt *Else = If->getElse();
      if (Else) {
        SmoothIfElseBodies.insert(Else);

        rewriter.InsertText(GET_LOC_BEFORE_END(Then), "/* ");
        rewriter.InsertText(Else->getBeginLoc(), "*/ ");

        assert(isa<CompoundStmt>(Else));
        rewriter.InsertText(Else->getBeginLoc().getLocWithOffset(1),
                              "\nsi_stack.enter_else(si_then_weights);");

        rewriter.InsertText(GET_LOC_BEFORE_END(Else), "\nsi_stack.exit_if_else();\n}");
      } else {
        rewriter.InsertText(GET_LOC_BEFORE_END(Then),
                            "\n{ /* else */"
                            "\nsi_stack.enter_else(si_then_weights);\n}\nsi_"
                            "stack.exit_if_else();\n}\n");
      }
    }

    if (smoothLoops) {
      if (isa<WhileStmt>(s)) {
        WhileStmt *While = cast<WhileStmt>(s);
        Expr *Cond = While->getCond();
        Stmt *Body = While->getBody();
        decorateLoop(s, While->getCond(), While->getBody());
        LoopBodies.insert(Body);
        return true;
      }

      if (isa<DoStmt>(s)) {
        DoStmt *Do = cast<DoStmt>(s);
        Expr *Cond = Do->getCond();
        Stmt *Body = Do->getBody();
        decorateLoop(s, Do->getCond(), Do->getBody(), true);
        LoopBodies.insert(Body);
        return true;
      }

      if (isa<ForStmt>(s)) {
        ForStmt *For = cast<ForStmt>(s);
        Expr *Cond = For->getCond();
        Stmt *Body = For->getBody();
        decorateLoop(s, For->getCond(), For->getBody());
        LoopBodies.insert(Body);
        return true;
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
        assert(!TypeStr.compare("void") || !TypeStr.compare("sdouble") || !TypeStr.compare("class sdouble")  ||
                                           !TypeStr.compare("adouble") || !TypeStr.compare("class adouble"));

        currSmoothFunctionEndLoc = GET_LOC_END(FuncBody);

        rewriter.InsertText(
            FuncBody->getBeginLoc().getLocWithOffset(1),
            "\nsi_stack.enter_scope();\n");

        if (!currSmoothFunctionIsVoid) {
          rewriter.InsertText(
              FuncBody->getBeginLoc().getLocWithOffset(1),
              "\nsdouble si_return_val;\n");
        }

        rewriter.InsertText(
            GET_LOC_BEFORE_END(FuncBody).getLocWithOffset(-1),
            "\nsi_stack.exit_function();\n");

        if (!currSmoothFunctionIsVoid) {
          rewriter.InsertText(
              GET_LOC_BEFORE_END(FuncBody).getLocWithOffset(-1),
              "return si_return_val.expectation();\n");
        }

        SmoothFuncBodies.insert(FuncBody);
     }
   }

    return true;
  }

  void decorateLoop(Stmt *stmt, Expr *Cond, Stmt *Body, bool isDoStmt = false) {
    assert(smoothLoops);

    rewriter.InsertText(stmt->getBeginLoc(), "si_stack.enter_loop();\n");
    rewriter.InsertText(GET_LOC_END(stmt), "si_stack.exit_loop();\n");
    rewriter.InsertText(GET_LOC_BEFORE_END(Body).getLocWithOffset(-1), "if (!si_stack.exit_loop_iteration())\nbreak;\n");

    if (!needsSmoothing(Cond))
      return;

    SmoothLoopBodies.insert(Body);

    auto CondText = rewriter.getRewrittenText(Cond->getSourceRange());
    rewriter.InsertText(Cond->getBeginLoc(), "true /* ");
    rewriter.InsertText(GET_LOC_BEFORE_END(Cond), " */");

    assert(isa<CompoundStmt>(Body));

    SourceLocation loc = isDoStmt ? GET_LOC_BEFORE_END(Body).getLocWithOffset(-1) : Body->getBeginLoc().getLocWithOffset(1);
    rewriter.InsertText(loc, "{ // check loop condition\n"
      "si_stack.prepare_branch();\n"
      "SiPathWeights si_then_weights = !(SiPathWeights)(" + CondText + ");\n"
      "{\n"
      "  si_stack.enter_if(si_then_weights);\n"
      "  si_stack.break_();\n"
      "}\n"
      "{\n"
      "  si_stack.enter_else(si_then_weights);\n"
      "}\n"
      "si_stack.exit_if_else(); }\n"
      "if (si_stack.top().empty())\n"
      "  break;\n");
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
