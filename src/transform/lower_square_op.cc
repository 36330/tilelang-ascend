#include <tvm/ir/transform.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <iostream>
#include <utility>

#include "../op/ascend.h"

namespace tvm {
namespace tl {

using namespace tir;

class LowerSquareOp : public StmtExprMutator {
 public:
  static PrimFunc Substitute(PrimFunc f) {
    LowerSquareOp mutator;
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = mutator.VisitStmt(f->body);
    return f;
  }

 private:
  Stmt VisitStmt_(const EvaluateNode* op) final {
    const CallNode* call = op->value.as<CallNode>();
    if (call && call->op.same_as(tl::ascend_square())) {
      ICHECK_EQ(call->args.size(), 3U);
      std::cerr << "[TileLang] LowerSquareOp rewrite tl.ascend_square -> tl.ascend_mul" << std::endl;

      Call new_call = Call(
          DataType::Handle(),
          tl::ascend_mul(),
          {
              call->args[0],  // dst
              call->args[1],  // src
              call->args[1],  // src again
              call->args[2],  // size
          });

      return Evaluate(new_call);
    }

    return StmtExprMutator::VisitStmt_(op);
  }
};

namespace transform {

tvm::transform::Pass LowerSquareOp() {
  auto pass_func = [](PrimFunc f, IRModule m,
                      tvm::transform::PassContext ctx) {
    return ::tvm::tl::LowerSquareOp::Substitute(std::move(f));
  };

  return tir::transform::CreatePrimFuncPass(
      pass_func,
      0,
      "tl.LowerSquareOp",
      {});
}

TVM_REGISTER_GLOBAL("tl.transform.LowerSquareOp")
    .set_body_typed(LowerSquareOp);

}  // namespace transform
}  // namespace tl
}  // namespace tvm

/*
before:
  Evaluate(Call(tl.ascend_square, [dst, src, size]))

after:
  Evaluate(Call(tl.ascend_mul, [dst, src, src, size]))
*/
