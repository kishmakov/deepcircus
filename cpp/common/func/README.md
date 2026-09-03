# cpp/common/func

`func.h` holds `Func`, the interface a boolean function of `bitness` bits
implements: call it on one input, or on a batch. `table.h` is the
truth-table-backed kind and its batch generation helpers. The code is in
namespace `func`; generator-dependent parts are compiled into `gen`, the rest
into `common`.
