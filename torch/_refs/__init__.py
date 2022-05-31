import torch

import torch._prims as prims
import torch._prims.utils as utils
from torch._prims.utils import (
    check,
    DimsType,
    ShapeType,
    StrideType,
    TensorLike,
    TensorLikeType,
    DeviceLikeType,
    TensorOrNumberLikeType,
    DimsSequenceType,
    TensorSequenceType,
    Number,
    NumberType,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    REDUCTION_OUTPUT_TYPE_KIND,
    is_weakly_lesser_type,
    dtype_to_type,
)
from torch._prims.wrappers import (
    elementwise_type_promotion_wrapper,
    out_wrapper,
    _maybe_convert_to_dtype,
    _maybe_resize_out,
    elementwise_unary_scalar_wrapper,
    _safe_copy_out,
)

from collections.abc import Iterable
from functools import reduce, partial
from typing import Sequence, Optional, Union, Callable, List, Tuple
import operator
import warnings
import math
from enum import Enum

# Experimental module containing prototype Python references for existing
#   PyTorch operations.

__all__ = [
    #
    # Elementwise Unary References
    #
    "abs",
    "acos",
    "acosh",
    "asin",
    "atan",
    "bitwise_not",
    # "cbrt",  # No corresponding torch operation
    "ceil",
    "cos",
    "cosh",
    "digamma",
    "erf",
    "erfinv",
    "erfc",
    "exp",
    "expm1",
    "exp2",
    "fill",
    "floor",
    "frac",
    "isfinite",
    "isinf",
    "isnan",
    "i0",
    "lgamma",
    "log",
    "log1p",
    "log2",
    "log10",
    "nan_to_num",
    "neg",
    "positive",
    "reciprocal",
    "round",  # TODO: model kwargs
    "sigmoid",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
    #
    # Elementwise Binary References
    #
    "add",
    "atan2",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    # "complex",
    # 'copysign', # where
    # 'div', # need to implement all rounding modes first
    "eq",
    "float_power",
    # 'floor_divide', # requires floor
    "fmax",
    "fmin",
    "fmod",
    # 'gcd',
    "ge",
    "gt",
    # 'heaviside',
    # 'hypot',
    "igamma",
    "igammac",
    "isclose",
    # 'lcm',
    # 'ldexp',
    "le",
    "logical_and",
    "logical_or",
    "logical_xor",
    "lt",
    # 'max', # implement with reductions
    "maximum",
    # 'min', # implement with reductions
    "minimum",
    "mul",
    "ne",
    "nextafter",
    # 'polar',  # abs, cos, sin
    "pow",
    # 'remainder',
    # 'rsub', # unblocked
    # # special.xlog1py
    # # special.zeta
    "sub",
    "true_divide",
    # 'xlogy', # where?, log, mul
    #
    # Elementwise Ternary References
    #
    "clamp",
    #
    # Conditional references
    #
    "where",
    #
    # Data conversion and movement references
    #
    "clone",
    "copy_to",  # TODO: add OpInfo (or implement .to)
    "item",  # TODO: add OpInfo
    #
    # Reduction ops
    #
    "all",
    "amax",
    "amin",
    "any",
    "mean",
    "std_mean",
    "std_var",
    "sum",
    "prod",
    "var",
    #
    # Linear algebra ops
    #
    "addr",
    #
    # View & Shape Ops
    #
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "as_strided",
    "broadcast_tensors",
    "cat",
    "chunk",
    "flatten",
    "flip",
    "fliplr",
    "flipud",
    "narrow",
    "permute",
    "reshape",
    "roll",
    "rot90",
    "stack",
    "swap_axes",  # alias for transpose
    "squeeze",
    "tensor_split",
    "transpose",
    "unsqueeze",
    "view",
    #
    # Tensor Creation
    #
    "empty",
    "empty_like",
    "empty_strided",
    "full",
    "full_like",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
    #
    # Randomness References
    #
    "uniform",  # TODO: add OpInfo -- and testing for randomness?
    #
    # Test-related functions
    #
    "equal",  # TODO: add OpInfo
]

Tensor = torch.Tensor


def _broadcast_shapes(*_shapes):
    shapes = tuple(filter(lambda x: x is not None, _shapes))

    # Short-circuits on no input
    if len(shapes) == 0:
        return None

    # Type checking
    # TODO: make common validations available as utils
    for shape in shapes:
        assert isinstance(shape, Sequence)

    # Computes common shape
    common_shape = [
        1,
    ] * reduce(max, (len(shape) for shape in shapes))
    for shape in shapes:
        for idx in range(-1, -1 - len(shape), -1):
            if common_shape[idx] == 1:
                if shape[idx] < 0:
                    raise ValueError(
                        "Attempting to broadcast a dimension with negative length!"
                    )
                common_shape[idx] = shape[idx]
            elif shape[idx] != 1:
                if common_shape[idx] != shape[idx]:
                    raise RuntimeError(
                        "Attempting to broadcast a dimension of length ",
                        str(shape[idx]),
                        "!",
                    )

    return common_shape


def _maybe_broadcast(*args, preserve_cpu_scalar_tensors=True):
    # Computes common shape
    common_shape = _broadcast_shapes(
        *map(lambda t: t.shape if isinstance(t, TensorLike) else None, args)
    )

    def __maybe_broadcast(x, shape):
        if x is None:
            return None
        elif isinstance(x, Number):
            return x
        elif isinstance(x, TensorLike):
            if preserve_cpu_scalar_tensors and utils.is_cpu_scalar_tensor(x):
                return x

            if tuple(x.shape) != common_shape:
                common_rank = len(common_shape) + 1
                start = common_rank - (len(x.shape) + 1)
                dims = tuple(range(start, len(x.shape) + start))
                return prims.broadcast_in_dim(x, common_shape, dims)
        else:
            raise RuntimeError(
                "Unexpected type when broadcasting: " + str(type(x)) + "!"
            )

    return tuple(__maybe_broadcast(x, common_shape) for x in args)


# Utilities should come BEFORE this import
from torch._decomp import register_decomposition

#
# Elementwise unary references
#

infer_aten_op = object()

# TODO: add type promotion support
def _make_elementwise_unary_reference(
    prim: Callable,
    *,
    type_promotion_kind,
    aten_op=infer_aten_op,
    disable_meta=False,
    extra_meta=None,
) -> Callable:
    @out_wrapper
    @elementwise_unary_scalar_wrapper
    @elementwise_type_promotion_wrapper(
        type_promoting_args=("a",),
        type_promotion_kind=type_promotion_kind,
    )
    def _ref(a: TensorLikeType) -> TensorLikeType:
        if not isinstance(a, TensorLike):
            raise RuntimeError(
                "Expected a tensor input for an elementwise unary operation!"
            )

        if extra_meta is not None:
            extra_meta(a)

        return prim(a)

    if aten_op is infer_aten_op:
        aten_op = getattr(torch.ops.aten, prim.__name__.split(".")[0])
    if aten_op is not None:
        register_decomposition(aten_op, disable_meta=disable_meta)(_ref)

    return _ref


abs = _make_elementwise_unary_reference(
    prims.abs,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)

acos = _make_elementwise_unary_reference(
    prims.acos,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

acosh = _make_elementwise_unary_reference(
    prims.acosh,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

asin = _make_elementwise_unary_reference(
    prims.asin,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

atan = _make_elementwise_unary_reference(
    prims.atan,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

bitwise_not = _make_elementwise_unary_reference(
    prims.bitwise_not,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)

ceil = _make_elementwise_unary_reference(
    prims.ceil,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)

cos = _make_elementwise_unary_reference(
    prims.cos,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

cosh = _make_elementwise_unary_reference(
    prims.cosh,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

digamma = _make_elementwise_unary_reference(
    prims.digamma,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

erf = _make_elementwise_unary_reference(
    prims.erf,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

erfinv = _make_elementwise_unary_reference(
    prims.erf_inv,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.erfinv,  # prim/aten name mismatch
)

erfc = _make_elementwise_unary_reference(
    prims.erfc,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

exp = _make_elementwise_unary_reference(
    prims.exp,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

expm1 = _make_elementwise_unary_reference(
    prims.expm1,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

exp2 = _make_elementwise_unary_reference(
    prims.exp2,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

# Fill has its own implementation because it has a value parameter
@out_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a,"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
)
def fill(a: TensorLikeType, value: NumberType) -> TensorLikeType:

    assert isinstance(a, TensorLike)
    assert isinstance(value, Number)

    python_type = utils.dtype_to_type(a.dtype)
    if not utils.is_weakly_lesser_type(type(value), python_type):
        msg = "value argument of type {0} cannot be safely cast to type {1}!".format(
            type(value), python_type
        )
        raise ValueError(msg)

    return prims.fill(a, value)


floor = _make_elementwise_unary_reference(
    prims.floor,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)


def _frac(x: TensorLikeType) -> TensorLikeType:
    trunc_x = mul(floor(abs(x)), sign(x))
    return sub(x, trunc_x)


frac = _make_elementwise_unary_reference(
    _frac,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.frac,
)


def _isfinite(a: TensorLikeType) -> TensorLikeType:
    if utils.is_float_dtype(a.dtype) or utils.is_complex_dtype(a.dtype):
        return prims.isfinite(a)

    return ones_like(a, dtype=torch.bool)


isfinite = _make_elementwise_unary_reference(
    _isfinite,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    aten_op=None,  # CompositeImplicitAutograd
)


def _isinf(a: TensorLikeType) -> TensorLikeType:
    # TODO Add complex tensor support to remove is_infinite prim
    # if utils.is_complex_dtype(a):
    #     return bitwise_or(_isinf(real(a), _isinf(imag(a))
    # else:
    #     return bitwise_not(bitwise_or(isnan(a), isfinite(a)))
    if utils.is_float_dtype(a.dtype) or utils.is_complex_dtype(a.dtype):
        return prims.is_infinite(a)

    return zeros_like(a, dtype=torch.bool)


isinf = _make_elementwise_unary_reference(
    _isinf,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    aten_op=torch.ops.aten.isinf,  # prim/aten name mismatch
)


def _isnan(a: TensorLikeType) -> TensorLikeType:
    return prims.ne(a, a)


isnan = _make_elementwise_unary_reference(
    _isnan,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    aten_op=torch.ops.aten.isnan,  # prim/aten name mismatch
)

i0 = _make_elementwise_unary_reference(
    prims.bessel_i0,
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.special_i0,
)

lgamma = _make_elementwise_unary_reference(
    prims.lgamma,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

log = _make_elementwise_unary_reference(
    prims.log,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

log1p = _make_elementwise_unary_reference(
    prims.log1p,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

log2 = _make_elementwise_unary_reference(
    prims.log2,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

log10 = _make_elementwise_unary_reference(
    prims.log10,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)


@register_decomposition(torch.ops.aten.nan_to_num)
@out_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a,"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def nan_to_num(
    a: TensorLikeType,
    *,
    nan: Optional[NumberType] = 0.0,
    posinf: Optional[NumberType] = None,
    neginf: Optional[NumberType] = None,
) -> TensorLikeType:
    assert isinstance(a, TensorLike)

    if a.dtype == torch.bool:
        return clone(a)

    if posinf is None:
        posinf = prims.maximum_value(a.dtype)

    if neginf is None:
        neginf = prims.minimum_value(a.dtype)

    result = where(isnan(a), nan, a)

    is_neg = signbit(a)
    is_neginf = bitwise_and(isinf(a), is_neg)
    result = where(is_neginf, neginf, result)

    is_posinf = bitwise_and(isinf(a), bitwise_not(is_neg))
    result = where(is_posinf, posinf, result)
    return result


def _neg_meta(a: TensorLikeType):
    if a.dtype is torch.bool:
        msg = "neg is not supported on bool tensors."
        raise RuntimeError(msg)


neg = _make_elementwise_unary_reference(
    prims.neg,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    extra_meta=_neg_meta,
)


# positive does not use _make_elementwise_unary_reference because it does not support out
def positive(a: TensorLikeType) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    if a.dtype is torch.bool:
        msg = "positive does not support bool tensors."
        raise RuntimeError(msg)
    return a


reciprocal = _make_elementwise_unary_reference(
    prims.reciprocal,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

# TODO: round takes additional kwargs
round = _make_elementwise_unary_reference(
    prims.round,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=None,  # TODO: this does need a decomp, but kwarg handling is needed
)


def _sigmoid(a: TensorLikeType) -> TensorLikeType:
    return true_divide(1, add(1, exp(neg(a))))


sigmoid = _make_elementwise_unary_reference(
    _sigmoid,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.sigmoid,
)

sign = _make_elementwise_unary_reference(
    prims.sign,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)


signbit = _make_elementwise_unary_reference(
    prims.signbit,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

sin = _make_elementwise_unary_reference(
    prims.sin,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

sinh = _make_elementwise_unary_reference(
    prims.sinh,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

sqrt = _make_elementwise_unary_reference(
    prims.sqrt,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

square = _make_elementwise_unary_reference(
    prims.square,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG,
    aten_op=None,  # CompositeImplicitAutograd,
)

tan = _make_elementwise_unary_reference(
    prims.tan,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)

tanh = _make_elementwise_unary_reference(
    prims.tanh, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)


def _make_elementwise_binary_reference(
    prim: Callable,
    *,
    type_promotion_kind,
    aten_op=infer_aten_op,
    has_out=True,
    supports_lhs_python_scalar=True,
    supports_rhs_python_scalar=True,
    disable_meta=False,
) -> Callable:
    @elementwise_type_promotion_wrapper(
        type_promoting_args=("a", "b"),
        type_promotion_kind=type_promotion_kind,
    )
    def _ref(
        a: Union[Tensor, NumberType],
        b: Union[Tensor, NumberType],
    ) -> Tensor:
        if not supports_lhs_python_scalar and isinstance(a, Number):
            raise ValueError(
                "Received a lhs Python scalar to an elementwise binary operation that does not accept lhs scalars!"
            )

        if not supports_rhs_python_scalar and isinstance(b, Number):
            raise ValueError(
                "Received a rhs Python scalar to an elementwise binary operation that does not accept rhs scalars!"
            )

        # TODO: enable this for operations that support it, like add
        if isinstance(a, Number) and isinstance(b, Number):
            raise ValueError(
                "Receive two Number inputs to an elementwise binary operation!"
            )

        a, b = _maybe_broadcast(a, b)
        return prim(a, b)

    if has_out:
        _ref = out_wrapper(_ref)

    if aten_op is infer_aten_op:
        aten_op = getattr(torch.ops.aten, prim.__name__.split(".")[0])
    if aten_op is not None:
        register_decomposition(aten_op, disable_meta=disable_meta)(_ref)

    return _ref


# Add has its own implementation because it has an alpha argument
@out_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def add(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
    *,
    alpha: Optional[NumberType] = None,
):
    """
    Reference implementation of torch.add
    """

    if isinstance(a, Number) and isinstance(b, Number):
        raise ValueError(
            "Receive two Number inputs to an elementwise binary operation!"
        )

    a, b = _maybe_broadcast(a, b)

    if alpha is not None:
        dtype = a.dtype if isinstance(a, TensorLike) else b.dtype  # type: ignore[union-attr]
        python_type = utils.dtype_to_type(dtype)
        if not utils.is_weakly_lesser_type(type(alpha), python_type):
            msg = (
                "alpha argument of type {0} cannot be safely cast to type {1}!".format(
                    type(alpha), python_type
                )
            )
            raise ValueError(msg)
        b = prims.mul(b, alpha)

    return prims.add(a, b)


# TODO: add docstring
atan2 = _make_elementwise_binary_reference(
    prims.atan2,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)

# TODO: add docstring
bitwise_and = _make_elementwise_binary_reference(
    prims.bitwise_and,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)

# TODO: add docstring
bitwise_left_shift = _make_elementwise_binary_reference(
    prims.shift_left,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.bitwise_left_shift,  # prim/aten name mismatch
)

# TODO: add docstring
bitwise_or = _make_elementwise_binary_reference(
    prims.bitwise_or,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)

# TODO: add docstring
bitwise_right_shift = _make_elementwise_binary_reference(
    prims.shift_right_arithmetic,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.bitwise_right_shift,  # prim/aten name mismatch
)

# TODO: add docstring
bitwise_xor = _make_elementwise_binary_reference(
    prims.bitwise_xor,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)

# TODO: add docstring
# complex =  _make_elementwise_binary_reference(prims.complex, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)

# TODO: add docstring
eq = _make_elementwise_binary_reference(
    prims.eq,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)

# TODO: add docstring
# Float power has its own implementation because it has unique type promotion.
# NB: aten_op not registered because CompositeExplicitAutograd
@out_wrapper
def float_power(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
) -> Tensor:

    if isinstance(a, Number) and isinstance(b, Number):
        raise ValueError(
            "Receive two Number inputs to an elementwise binary operation!"
        )

    # Handles type promotion
    dtype = utils.get_higher_dtype(a, b)
    assert dtype is not None
    if utils.is_complex_dtype(dtype):
        dtype = torch.complex128
    else:
        dtype = torch.float64

    # Float power has the following contiguous cast behavior to be
    # consistent with its C++ impl
    if isinstance(a, TensorLike) and a.dtype != dtype:
        a = prims.to_dtype(a, dtype)
    if isinstance(b, TensorLike) and b.dtype != dtype:
        b = prims.to_dtype(b, dtype)

    a, b = _maybe_broadcast(a, b)
    return prims.pow(a, b)


# TODO: add docstring
fmax = _make_elementwise_binary_reference(
    prims.fmax,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.fmax,
)

# TODO: add docstring
fmin = _make_elementwise_binary_reference(
    prims.fmin,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.fmin,
)

# TODO: add docstring
fmod = _make_elementwise_binary_reference(
    prims.fmod,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.fmod,
)

# TODO: add docstring
ge = _make_elementwise_binary_reference(
    prims.ge,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)

# TODO: add docstring
gt = _make_elementwise_binary_reference(
    prims.gt,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)

igamma = _make_elementwise_binary_reference(
    prims.igamma,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)

igammac = _make_elementwise_binary_reference(
    prims.igammac,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)


def isclose(
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> TensorLikeType:
    if a.dtype != b.dtype:
        msg = "Attempting to compare tensors of different dtypes {0} and {1}!".format(
            a.dtype, b.dtype
        )
        raise ValueError(a, b)
    if rtol < 0:
        msg = "rtol must be greater than or equal to zero, but got {0}!".format(rtol)
    if atol < 0:
        msg = "atol must be greater than or equal to zero, but got {0}!".format(atol)

    close = eq(a, b)
    if equal_nan and (utils.is_float_dtype(a.dtype) or utils.is_complex_dtype(a.dtype)):
        close = logical_or(close, logical_and(isnan(a), isnan(b)))

    # Note: In case of zero tolerances the closeness inequality degenerates to an equality check.
    # In this case, the short-circuit prevents false positives as detailed in the paragraph below.
    if atol == 0 and rtol == 0:
        return close

    # Note [closeness error computation]
    # atol and rtol are provided as doubles, so the computation
    # rtol * other will produce a float or complex tensor.
    # When the difference (self - other) is compared to it then the
    # tensor representing the difference will also be cast to float or complex.
    # However, since (self - other) in uint8 is very likely to produce a
    # negative value, this moves the cast forward so the difference is
    # always computed in a float or complex type.
    # If the values of the integer tensors cannot be exactly represented
    # by the default scalar type then this may cause an incorrect result.
    if not utils.is_float_dtype(a.dtype) and not utils.is_complex_dtype(a.dtype):
        a = prims.convert_element_type(a, torch.get_default_dtype())
        b = prims.convert_element_type(b, torch.get_default_dtype())

    allowed_error = add(atol, abs(mul(b, rtol)))
    actual_error = abs(sub(a, b))

    # Computes finite closeness
    result = logical_or(
        close, logical_and(isfinite(actual_error), le(actual_error, allowed_error))
    )

    return result


# TODO: add docstring
le = _make_elementwise_binary_reference(
    prims.le,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)


def _logical_and(a: TensorLikeType, b: TensorLikeType):
    if not utils.is_boolean_dtype(a.dtype):
        a = ne(a, 0)
    if not utils.is_boolean_dtype(b.dtype):
        b = ne(b, 0)
    return bitwise_and(a, b)


logical_and = _make_elementwise_binary_reference(
    _logical_and,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    aten_op=torch.ops.aten.logical_and,
)


def _logical_or(a: TensorLikeType, b: TensorLikeType):
    if not utils.is_boolean_dtype(a.dtype):
        a = ne(a, 0)
    if not utils.is_boolean_dtype(b.dtype):
        b = ne(b, 0)
    return bitwise_or(a, b)


logical_or = _make_elementwise_binary_reference(
    _logical_or,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    aten_op=torch.ops.aten.logical_or,
)


def _logical_xor(a: TensorLikeType, b: TensorLikeType):
    if not utils.is_boolean_dtype(a.dtype):
        a = ne(a, 0)
    if not utils.is_boolean_dtype(b.dtype):
        b = ne(b, 0)
    return bitwise_xor(a, b)


# TODO: skip unnecessary conversion of long to float
logical_xor = _make_elementwise_binary_reference(
    _logical_xor,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    aten_op=torch.ops.aten.logical_xor,
)


# TODO: add docstring
lt = _make_elementwise_binary_reference(
    prims.lt,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)

# TODO: add docstring
maximum = _make_elementwise_binary_reference(
    prims.maximum,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)

# TODO: add docstring
minimum = _make_elementwise_binary_reference(
    prims.minimum,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)

# TODO: add docstring
mul = _make_elementwise_binary_reference(
    prims.mul,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)

# TODO: add docstring
ne = _make_elementwise_binary_reference(
    prims.ne,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)

# TODO: add docstring
nextafter = _make_elementwise_binary_reference(
    prims.nextafter,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)

# TODO: add docstring
pow = _make_elementwise_binary_reference(
    prims.pow,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG,
)

# TODO: add docstring
# TODO: consider refactoring this with add impl
# sub has its own implementation because it has an alpha argument
@register_decomposition(torch.ops.aten.sub)
@out_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def sub(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
    *,
    alpha: Optional[NumberType] = None,
):
    """
    Reference implementation of torch.add
    """

    if isinstance(a, Number) and isinstance(b, Number):
        raise ValueError(
            "Receive two Number inputs to an elementwise binary operation!"
        )

    a, b = _maybe_broadcast(a, b)

    if alpha is not None:
        dtype = a.dtype if isinstance(a, TensorLike) else b.dtype  # type: ignore[union-attr]
        python_type = utils.dtype_to_type(dtype)
        if not utils.is_weakly_lesser_type(type(alpha), python_type):
            msg = (
                "alpha argument of type {0} cannot be safely cast to type {1}!".format(
                    type(alpha), python_type
                )
            )
            raise ValueError(msg)
        b = prims.mul(b, alpha)

    return prims.sub(a, b)


# TODO: add docstring
true_divide = _make_elementwise_binary_reference(
    prims.div,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=None,  # CompositeImplicitAutograd
)

#
# Elementwise Ternary References
#


@out_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "min", "max"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def clamp(
    a: TensorLikeType,
    min: Optional[TensorOrNumberLikeType] = None,
    max: Optional[TensorOrNumberLikeType] = None,
) -> TensorLikeType:
    a, min, max = _maybe_broadcast(a, min, max)

    if min is not None and max is not None:
        return minimum(maximum(a, min), max)
    if min is not None:
        return maximum(a, min)
    if max is not None:
        return minimum(a, max)

    msg = "clamp called but both min and max are none!"
    raise ValueError(msg)


#
# Conditional references
#

# https://pytorch.org/docs/stable/generated/torch.where.html
# TODO: implement alternate where
@register_decomposition(torch.ops.aten.where)
@out_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
)
def where(
    pred: Tensor,
    a: Optional[Union[TensorLikeType, NumberType]] = None,
    b: Optional[Union[TensorLikeType, NumberType]] = None,
):
    """ """

    if a is None or b is None:
        raise NotImplementedError

    utils.check_same_device(pred, a, b, allow_cpu_scalar_tensors=True)
    assert pred.dtype is torch.bool

    pred, a, b = _maybe_broadcast(pred, a, b)
    return prims.where(pred, a, b)


#
# Data Movement References
#
def clone(
    a: TensorLikeType, *, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:

    return prims.clone(a, memory_format=memory_format)


def copy_to(a: Tensor, b: Tensor, *, allow_cross_device=True):
    if not allow_cross_device and a.device != b.device:
        msg = "Attempting to copy from device {0} to device {1}, but cross-device copies are not allowed!".format(
            b.device, a.device
        )
        raise RuntimeError(msg)

    return prims.copy_to(a, b)


def item(a: TensorLikeType) -> NumberType:
    if a.numel() != 1:
        msg = f"Can't convert a tensor with {a.numel()} elements to a number!"
        raise ValueError(msg)

    # NOTE: explicit conversion is necessary for bool!
    # See https://github.com/pytorch/pytorch/issues/78071
    number_type = utils.dtype_to_type(a.dtype)
    return number_type(prims.item(a))


#
# Reduction references
#


def _reduction(
    a: TensorLikeType,
    prim: Callable,
    *,
    has_identity: bool = True,
    accepts_dim_tuple: bool = True,  # to handle min/argmin that accept single dim only
    dims: Optional[DimsType] = None,
    keepdims: bool = False,
    dtype: Optional[torch.dtype] = None,  # should be specified for ops that support it
    out: Optional[Tensor] = None,
    output_dtype_kind: REDUCTION_OUTPUT_TYPE_KIND,
) -> TensorLikeType:  # it is usually SAME, but I want
    # ref writers to actually think about what to put here
    assert isinstance(a, TensorLike)
    if a.ndim > 64:
        raise RuntimeError(
            "Received a tensor with {0} dimensions, but only tensors with up to 64 dims are supported!".format(
                a.ndim
            )
        )

    if out is not None:
        assert isinstance(out, TensorLike)
        if dtype is not None:
            # TODO - this is true for eager mode currently, but it's wrong behavior for complex norms
            if dtype != out.dtype:
                raise RuntimeError(
                    "dtype argument and out dtype must match in reduction"
                )
    if not accepts_dim_tuple:
        assert dims is None or isinstance(dims, int)
    if isinstance(dims, int):
        dims = (dims,)  # type: ignore[assignment]
    dims = utils.reduction_dims(a.shape, dims)
    if not has_identity:
        valid_shape = a.ndim == 0 or py_all(a.shape[i] for i in dims)
        if not valid_shape:
            raise RuntimeError(
                "reducing over zero-size dimension for reduction operation without identity"
            )
    computation_dtype, result_dtype = utils.reduction_dtypes(
        a, output_dtype_kind, dtype
    )
    a_converted = prims.convert_element_type(a, computation_dtype)
    result = prim(a_converted, dims)
    if keepdims:
        output_shape = [a.shape[i] if i not in dims else 1 for i in range(a.ndim)]
        broadcast_dims = [i for i in range(a.ndim) if i not in dims]
        result = prims.broadcast_in_dim(result, output_shape, broadcast_dims)

    if out is not None:
        assert result_dtype is not None
        if dtype is not None and result_dtype != out.dtype:
            raise RuntimeError(
                "Expected the dtype of reduction result and out to match"
            )
        out = _maybe_resize_out(out, result.shape)
        return _safe_copy_out(copy_from=result, copy_to=out)  # type: ignore[arg-type]

    if result.dtype != result_dtype and result_dtype is not None:
        result = prims.convert_element_type(result, result_dtype)

    return result


# Saves Python all
py_all = all


@out_wrapper
def all(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
) -> TensorLikeType:
    # Computes nelem
    if isinstance(dim, int):
        dim = (dim,)  # type: ignore[assignment]
    dims = utils.reduction_dims(a.shape, dim)  # type: ignore[arg-type]
    nelem = 1 if a.ndim == 0 else reduce(operator.mul, (a.shape[i] for i in dims), 1)

    a_ = _maybe_convert_to_dtype(a, torch.bool)
    result = eq(sum(a_, dim=dim, keepdim=keepdim), nelem)  # type: ignore[arg-type]

    # Preserves uint8 -- probably a legacy mask thing
    if a.dtype is torch.uint8:
        return prims.convert_element_type(result, torch.uint8)

    return result


@out_wrapper
def any(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
) -> TensorLikeType:
    a_ = _maybe_convert_to_dtype(a, torch.bool)
    result = ne(sum(a_, dim=dim, keepdim=keepdim), False)  # type: ignore[arg-type]

    # Preserves uint8 -- probably a legacy mask thing
    if a.dtype is torch.uint8:
        return prims.convert_element_type(result, torch.uint8)

    return result


@register_decomposition(torch.ops.aten.sum)
def sum(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    keepdim: bool = False,
    *,
    dtype=None,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    if dtype is None:
        if utils.is_boolean_dtype(a.dtype) or utils.is_integer_dtype(a.dtype):
            dtype = torch.int64
        else:
            dtype = a.dtype
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None
    return _reduction(
        a,
        prims.sum,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        out=out,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


@register_decomposition(torch.ops.aten.prod)
def prod(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    keepdim: bool = False,
    *,
    dtype=None,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    if dtype is None:
        if utils.is_boolean_dtype(a.dtype) or utils.is_integer_dtype(a.dtype):
            dtype = torch.int64
        else:
            dtype = a.dtype
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None
    return _reduction(
        a,
        prims.prod,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        out=out,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


def amin(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    keepdim: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None

    return _reduction(
        a,
        prims.amin,
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        out=out,
        has_identity=False,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


def amax(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    keepdim: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None

    return _reduction(
        a,
        prims.amax,
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        out=out,
        has_identity=False,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


def _set_correction(
    unbiased: Optional[bool] = None,
    correction: Optional[int] = None,
):
    if correction is not None and unbiased is not None:
        raise RuntimeError("cannot specify both correction and unbiased arguments")
    elif correction is None and unbiased is None:
        correction = 1
    elif correction is None and unbiased is not None:
        correction = 0 if unbiased is False else 1
    if not isinstance(correction, int):
        raise ValueError("correction argument should be integer")
    if correction < 0:
        raise ValueError("correction argument should be non-negative")
    return correction


@out_wrapper
def var(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[int] = None,
) -> TensorLikeType:
    correction = _set_correction(unbiased, correction)
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None

    result = _reduction(
        a,
        partial(prims.var, correction=correction),
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        out=None,
        has_identity=True,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    )
    return result


@out_wrapper
def std(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[int] = None,
) -> TensorLikeType:
    correction = _set_correction(unbiased, correction)
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None

    opmath_dtype, dtype = utils.reduction_dtypes(
        a, REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT
    )

    result = _reduction(
        a,
        partial(prims.var, correction=correction),
        dims=dim,
        keepdims=keepdim,
        dtype=opmath_dtype,
        out=None,
        has_identity=True,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    )
    result = sqrt(result)
    return _maybe_convert_to_dtype(result, dtype)  # type: ignore[return-value,arg-type]


def mean(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    keepdim: bool = False,
    *,
    dtype=None,
    out=None,
) -> TensorLikeType:
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None
    if dtype is None:
        dtype = a.dtype
    # can't use out wrapper because of this argument
    if out is not None and out.dtype != dtype:
        raise RuntimeError("expected out dtype and dtype to match")
    result = _reduction(
        a,
        prims.sum,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        out=None,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.KEEP_PROMOTED_TYPE,
    )
    if utils.is_integer_dtype(dtype):
        raise RuntimeError("result type should be floating point or complex")
    if isinstance(dim, int):
        dim = (dim,)  # type: ignore[assignment]
    dims = utils.reduction_dims(a.shape, dim)  # type: ignore[arg-type]
    nelem = 1 if a.ndim == 0 else reduce(operator.mul, (a.shape[i] for i in dims), 1)
    result = true_divide(result, nelem)
    result_dtype = a.dtype if dtype is None else dtype
    result = _maybe_convert_to_dtype(result, result_dtype)  # type: ignore[assignment]
    if out is not None:
        assert isinstance(out, TensorLike)
        out = _maybe_resize_out(out, result.shape)
        return _safe_copy_out(copy_from=result, copy_to=out)  # type: ignore[arg-type]
    return result


@register_decomposition(torch.ops.aten.std_mean.correction)
def std_mean(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[int] = None,
):
    s = std(a, dim, unbiased, keepdim, correction=correction)
    m = mean(a, dim, keepdim)
    return s, m


def var_mean(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[int] = None,
):
    v = var(a, dim, unbiased, keepdim, correction=correction)
    m = mean(a, dim, keepdim)
    return v, m


@register_decomposition(torch.ops.aten.addr)
@out_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "vec1", "vec2"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def addr(
    self: TensorLikeType,
    vec1: TensorLikeType,
    vec2: TensorLikeType,
    beta: NumberType = 1,
    alpha: NumberType = 1,
) -> TensorLikeType:
    check(
        vec1.ndim == 1,
        lambda: f"addr: Expected 1-D argument vec1, but got {vec1.ndim}-D",
    )
    check(
        vec2.ndim == 1,
        lambda: f"addr: Expected 1-D argument vec2, but got {vec2.ndim}-D",
    )
    self = self.expand(vec1.shape[0], vec2.shape[0])
    if utils.is_boolean_dtype(self.dtype):
        # Integers are accepted for booleans
        check(
            is_weakly_lesser_type(type(beta), int),
            f"expected bool/int beta but got {type(beta)}",
        )
        check(
            is_weakly_lesser_type(type(alpha), int),
            f"expected bool/int alpha but got {type(beta)}",
        )
        if not beta:
            return torch.outer(vec1, vec2) if alpha else torch.full_like(self, False)
        else:
            return torch.logical_or(
                self,
                torch.outer(vec1, vec2) if alpha else torch.full_like(self, False),
            )
    else:
        check(
            is_weakly_lesser_type(type(beta), dtype_to_type(self.dtype)),
            f"cannot safely convert {type(beta)} to {self.dtype}",
        )
        check(
            is_weakly_lesser_type(type(alpha), dtype_to_type(self.dtype)),
            f"cannot safely convert {type(alpha)} to {self.dtype}",
        )
        if beta == 0:
            # This means NaNs from self are dropped if beta is zero
            return alpha * torch.outer(vec1, vec2)
        else:
            return beta * self + alpha * torch.outer(vec1, vec2)


def atleast_1d(
    *args: TensorLikeType,
) -> Union[TensorLikeType, Tuple[TensorLikeType, ...]]:
    """Reference implementation of :func:`torch.atleast_1d`."""
    args_ = args[0] if len(args) == 1 and not torch.is_tensor(args[0]) else args
    res = tuple(a if a.ndim >= 1 else unsqueeze(a, 0) for a in args_)
    return res if len(res) > 1 else res[0]


# Helper function with assert to avoid MyPy error
# of incompatible type passed to unsqueeze
def _unsqueeze_atleast(
    at_least_fn: Callable, dim: int, arg: TensorLikeType
) -> TensorLikeType:
    arg_ = at_least_fn(arg)
    assert isinstance(arg_, TensorLike)
    return unsqueeze(arg_, dim)


def atleast_2d(
    *args: TensorLikeType,
) -> Union[TensorLikeType, Tuple[TensorLikeType, ...]]:
    """Reference implementation of :func:`torch.atleast_2d`."""
    args_ = args[0] if len(args) == 1 and not torch.is_tensor(args[0]) else args
    unsqueeze_atleast_1d = partial(_unsqueeze_atleast, atleast_1d, 0)
    res = tuple(a if a.ndim >= 2 else unsqueeze_atleast_1d(a) for a in args_)
    return res if len(res) > 1 else res[0]


def atleast_3d(
    *args: TensorLikeType,
) -> Union[TensorLikeType, Tuple[TensorLikeType, ...]]:
    """Reference implementation of :func:`torch.atleast_3d`."""
    args_ = args[0] if len(args) == 1 and not torch.is_tensor(args[0]) else args
    unsqueeze_atleast_2d = partial(_unsqueeze_atleast, atleast_2d, -1)
    res = tuple(a if a.ndim >= 3 else unsqueeze_atleast_2d(a) for a in args_)
    return res if len(res) > 1 else res[0]


def as_strided(
    a: TensorLikeType, size: ShapeType, stride: StrideType, storage_offset: int = 0
) -> TensorLikeType:
    return prims.as_strided(a, size, stride, storage_offset)


def broadcast_tensors(*tensors) -> List[TensorLikeType]:
    return list(_maybe_broadcast(*tensors, preserve_cpu_scalar_tensors=False))


@out_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("tensors",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
)
def cat(tensors: TensorSequenceType, dim: int = 0) -> TensorLikeType:
    if len(tensors) == 0:
        msg = "cat expects at least one tensor, but received zero!"
        raise ValueError(msg)

    for tensor in tensors:
        assert isinstance(tensor, TensorLike)

    utils.check_same_device(*tensors, allow_cpu_scalar_tensors=False)

    dim = utils.canonicalize_dim(tensors[0].ndim, dim)
    utils.validate_idx(tensors[0].ndim, dim)

    # Filters tensors with one dimension of length zero
    filtered = tuple(x for x in tensors if not (x.ndim == 1 and x.numel() == 0))
    if len(filtered) == 0:
        t = tensors[0]

        # TODO: fix this to work with meta tensors
        try:
            requires_grad = any(x.requires_grad for x in tensors)
        except Exception:
            requires_grad = False

        return empty((0,), dtype=t.dtype, device=t.device, requires_grad=requires_grad)

    return prims.cat(filtered, dim)


def chunk(a: TensorLikeType, chunks: int, dim: int = 0) -> Tuple[TensorLikeType, ...]:
    if chunks <= 0:
        msg = "Expected at least one chunk, but got {0}!".format(chunks)
        raise ValueError(msg)

    dim = utils.canonicalize_dim(a.ndim, dim)
    length = a.shape[dim]
    chunk_size = math.ceil(length / chunks)
    full_chunks = math.floor(length / chunk_size)
    tail_chunk_size = length % chunk_size

    result = []
    for i in range(full_chunks):
        result.append(narrow(a, dim, i * chunk_size, chunk_size))

    if tail_chunk_size != 0:
        result.append(narrow(a, dim, full_chunks * chunk_size, tail_chunk_size))

    return tuple(result)


# Note: flatten, unlike prim.collapse and prim.collapse_view has an inclusive end_dim
# Note: flatten, unlike other shape operators, returns the input tensor on a no-op (unless
# a 0D tensor is flattened, in which case it's returned in 1D)
def flatten(a: TensorLikeType, start_dim: int = 0, end_dim: int = -1) -> TensorLikeType:
    start_dim = utils.canonicalize_dim(a.ndim, start_dim)
    end_dim = utils.canonicalize_dim(a.ndim, end_dim)

    # Short-circuits on no-op
    if start_dim == end_dim and a.ndim != 0:
        return a

    # Tries to take a view
    # TODO: we could look at directing collapse_view to skip its meta function here (unsafe_collapse_view)
    new_shape, new_strides = prims._collapse_view_helper(a, start_dim, end_dim + 1)
    if new_shape is not None:
        return prims.collapse_view(a, start_dim, end_dim + 1)

    # Makes a copy if it can't make a view
    return prims.collapse(a, start_dim, end_dim + 1)


@register_decomposition(torch.ops.aten.flip)
def flip(a: TensorLikeType, dims: DimsSequenceType) -> TensorLikeType:
    if not isinstance(dims, tuple) and not isinstance(dims, list):
        raise ValueError("dims has to be a sequence of ints")
    dims = utils.canonicalize_dims(a.ndim, dims)  # type: ignore[assignment]
    utils.validate_no_repeating_dims(dims)
    return prims.rev(a, dims)


def fliplr(a: TensorLikeType) -> TensorLikeType:
    if a.ndim < 2:
        raise RuntimeError("Input must be >= 2-d.")

    return flip(a, (1,))


def flipud(a: TensorLikeType) -> TensorLikeType:
    if a.ndim < 1:
        raise RuntimeError("Input must be >= 1-d.")

    return flip(a, (0,))


def narrow(a: TensorLikeType, dim: int, start: int, length: int) -> TensorLikeType:
    dim = utils.canonicalize_dim(a.ndim, dim)
    return prims.slice_in_dim(a, start, start + length, axis=dim)


def permute(a: TensorLikeType, dims: DimsSequenceType) -> TensorLikeType:
    _permutation = utils.canonicalize_dims(a.ndim, dims)
    return prims.transpose(a, _permutation)


def _reshape_view_helper(
    a: TensorLikeType, shape: ShapeType, *, allow_copy: bool
) -> TensorLikeType:
    # NOTE: Reshape may be given a shape with a -1 length
    # This indicates that the dimension's length should be inferred
    # Creates a valid shape

    for idx in range(len(shape)):
        if shape[idx] == -1:
            # Verifies there's only one dimension of length -1 in the shape
            if shape.count(-1) > 1:
                msg = "Can only infer the length of one dimension, but got shape {0}!".format(
                    str(shape)
                )
                raise ValueError(msg)

            # TODO: improve error message
            if a.numel() > 0:
                length = reduce(
                    operator.floordiv, (x for x in shape if x != -1), a.numel()
                )
            else:
                msg = "Cannot reshape a tensor of zero elements into shape {0} because the unspecified length is ambiguous!".format(
                    str(shape)
                )
                raise ValueError(msg)

            shape = list(shape)
            shape[idx] = length
            break

    # Short-circuits if shape is the same
    utils.validate_shape(shape)
    if tuple(a.shape) == tuple(shape):
        return prims.view_of(a)

    numel = reduce(operator.mul, shape) if len(shape) > 0 else 1
    if a.numel() != numel:
        msg = "Attempting to reshape a tensor with shape {0} and {1} elements to a shape {2} with {3} elements!".format(
            str(a.shape), a.numel(), str(shape), numel
        )
        raise ValueError(msg)

    # Special-cases tensors with no elements
    if a.numel() == 0:
        return as_strided(a, shape, utils.make_contiguous_strides_for(shape))

    # Special-cases reshaping zero dim tensors
    if a.ndim == 0:
        _a = a
        for length in shape:
            assert length == 1
            _a = unsqueeze(_a, -1)
        return _a

    # Special-cases reshaping to zero dim tensors
    if len(shape) == 0:
        _a = a
        for length in a.shape:
            assert length == 1
            _a = squeeze(_a, -1)
        return _a

    # Handles general case: a 1+D tensor reshaped into a distinct 1+D shape

    # NOTE [Reshape Algorithm]
    # This algorithm works by attempting to greedily construct the desired dimensions in
    # the output shape, left to right. It does this by, conceptually, accumulating
    # dimensions of the original tensor, also left to right, until the dimension
    # can be constructed using prims.split_dim.
    # The algorithm also has special handling for tail squeezes/unsqueezes, like
    # if a reshape from (5, 5) to (5, 5, 1) or vice versa.
    #
    # This algorithm does not flatten the original tensor and then split dims as appropriate
    # because that would create copies more often than this algorithm. flatten is the only
    # operation below which can create a view or a copy, and while it prefers creating
    # views it may sometimes create a copy if the tensor's strides do not permit a view.
    # As a result, this algorithm tries to minimize flattening.
    #
    # Note that a better version of this algorithm may exist. Regions which could be
    # flattened without creating a copy can be identified in advance, and that might
    # allow fewer flatten calls or faster short-circuiting to make a copy.
    idx = 0
    a_ = a
    for length in shape:
        # Handles tail unsqueezes
        if idx >= a_.ndim:
            assert length == 1
            last_dim = a_.ndim - 1
            # NOTE: using split_dim instead of unsqueeze may seem silly here,
            # but it's necessary to get the strides correct
            a_ = prims.split_dim(a_, last_dim, a_.shape[last_dim])
            idx = idx + 1
            continue

        # Skips dimensions that are already the correct length
        if length == a_.shape[idx]:
            idx = idx + 1
            continue

        # Gathers enough original dimensions such that this new dimension can be created
        # Note that this accumulation will terminate because we've verified a and the shape
        # specify the same number of elements above
        accum = a_.shape[idx]
        end = idx
        while accum % length != 0:
            end = end + 1
            accum = accum * a_.shape[end]
        if end != idx:
            # NOTE: in this case multiple dimensions must be flatten to create the desired dimension
            # This flattening is why reshape sometimes creates a copy -- because flattening
            # may return a view of a copy

            # Checks if collapse can be a view and short-circuits to copying reshape if it can't
            new_shape, new_strides = prims._collapse_view_helper(a_, idx, end + 1)
            if new_shape is None:
                if allow_copy:
                    return prims.reshape(a, shape)

                msg = "Cannot view a tensor with shape {0} and strides {1} as a tensor with shape {2}!".format(
                    a.shape, a.stride(), shape
                )
                raise ValueError(msg)

            a_ = flatten(a_, idx, end)

        # Splits the (possibly flattened) dimension to create the desired dim length
        if accum != length:
            a_ = prims.split_dim(a_, idx, length)

        idx = idx + 1

    # Squeezes tail
    while idx < a_.ndim:
        assert a_.shape[idx] == 1
        a_ = squeeze(a_, idx)

    return a_


def reshape(a: TensorLikeType, shape: ShapeType) -> TensorLikeType:
    return _reshape_view_helper(a, shape, allow_copy=True)


def roll(
    a: TensorLikeType, shifts: DimsType, dims: DimsType = tuple()
) -> TensorLikeType:
    """Reference implementation of :func:`torch.roll`."""
    dims = utils.canonicalize_dims(a.ndim, dims)
    # ATen specifies int[1] type for shifts and dims which expands integers to tuples of length 1
    if not isinstance(shifts, Iterable):
        shifts = (shifts,)
    if not isinstance(dims, Iterable):
        dims = (dims,)

    # Avoid modulo by zero
    if a.numel() == 0:
        return clone(a)

    len_shifts = len(shifts)
    len_dims = len(dims)
    if len_shifts != 1 or len_dims != 1:
        if len_shifts == 0:
            raise RuntimeError("`shifts` required")
        # Takes care of the case when dims is not specified (default)
        # By default, the tensor is flattened before shifting, after which the original shape is restored
        if len_dims == 0 and len_shifts == 1:
            return view(roll(flatten(a), shifts, 0), a.shape)
        if len_shifts != len_dims:
            raise RuntimeError(
                f"shifts and dimensions must align. shifts: {len_shifts}, dims: {len_dims}"
            )
        assert len_dims > 1
        tail_shifts = shifts[1:]
        tail_dims = dims[1:]
        first_dim_rolled = roll(a, shifts[0], dims[0])
        return roll(first_dim_rolled, tail_shifts, tail_dims)

    # This path is taken when only one dimension is rolled
    # For example to get `first_dim_rolled` above
    dim = dims[0]
    size = a.shape[dim]
    start = (size - shifts[0]) % size
    t0 = narrow(a, dim, start, size - start)
    t1 = narrow(a, dim, 0, start)
    return cat((t0, t1), dim)


def rot90(
    a: TensorLikeType, k: int = 1, dims: DimsSequenceType = (0, 1)
) -> TensorLikeType:
    """Reference implementation of :func:`torch.rot90`."""
    dims_ = utils.canonicalize_dims(a.ndim, dims)
    # Required to silence MyPy errors
    assert isinstance(dims_, (tuple, list))
    dims = dims_
    if len(dims) != 2:
        raise RuntimeError(
            f"expected total rotation dims == 2, but got dims = {len(dims)}"
        )
    if a.ndim < 2:
        raise RuntimeError(f"expected total dims >= 2, but got total dims = {a.ndim}")
    if dims[0] == dims[1]:
        raise RuntimeError(
            f"expected rotation dims to be different, but got dim0 = {dims[0]} and dim1 = {dims[1]}"
        )
    k = k % 4  # Rotation direction is from the second towards the first axis for k < 0
    if k == 1:
        return transpose(flip(a, (dims[1],)), dims[0], dims[1])
    elif k == 2:
        return flip(a, dims)
    elif k == 3:
        return transpose(flip(a, (dims[0],)), dims[0], dims[1])
    else:
        return clone(a)


# update to cat then view instead of unsqueezing each tensor
@out_wrapper
def stack(tensors: TensorSequenceType, dim: int = 0) -> TensorLikeType:
    tensors = tuple(unsqueeze(a, dim) for a in tensors)
    return cat(tensors, dim)


# Note: although squeeze is documented as having the out= kwarg it doesn't
def squeeze(a: TensorLikeType, dim: Optional[int] = None) -> TensorLikeType:
    if dim is not None:
        dim = utils.canonicalize_dim(a.ndim, dim)
        # Short-circuits if the tensor has no dimensions
        if len(a.shape) == 0:
            assert dim == 0
            return prims.view_of(a)

        # Note: squeeze does not modify tensors when the given dim is not a dimension of length 1
        if a.shape[dim] != 1:
            return prims.view_of(a)
        return prims.squeeze(a, (dim,))

    dims = tuple(idx for idx in range(len(a.shape)) if a.shape[idx] == 1)
    return prims.squeeze(a, dims)


# Note: does not work with TensorMetas because of data-dependent control-flow
def tensor_split(
    a: TensorLikeType,
    indices_or_sections: Union[Tensor, DimsType],
    dim: int = 0,
) -> Tuple[TensorLikeType, ...]:
    _dim = utils.canonicalize_dim(a.ndim, dim)
    if a.ndim == 0:
        msg = "tensor_split: received a rank zero tensor, but expected a tensor of rank one or greater!"
        raise ValueError(msg)

    # If indices_or_sections is a tensor, it must be a CPU Long tensor
    if isinstance(indices_or_sections, TensorLike):
        if indices_or_sections.device != torch.device("cpu"):
            msg = "tensor_split: if indices_or_sections is a tensor it must be on the CPU, but received one on {0}".format(
                indices_or_sections.device
            )
            raise ValueError(msg)
        if indices_or_sections.dtype != torch.long:
            msg = "tensor_split: if indices_or_sections is a tensor it must have long dtype, "
            " but received one with dtype {0}".format(indices_or_sections.dtype)
            raise ValueError(msg)

    # Case 0 -- indices_or_sections is an integer or a scalar tensor n and a is split along dim into n parts of equal-ish length
    if isinstance(indices_or_sections, int) or (
        isinstance(indices_or_sections, TensorLike) and indices_or_sections.ndim == 0
    ):
        sections: int = (
            indices_or_sections  # type: ignore[assignment]
            if isinstance(indices_or_sections, Number)
            else indices_or_sections.item()
        )

        if sections <= 0:
            msg = "tensor_split: number of sections must be greater than 0, but was {0}".format(
                sections
            )
            raise ValueError(msg)

        splits = []
        dim_size = a.shape[_dim]
        min_split_size = math.floor(dim_size / sections)
        num_splits_one_extra = dim_size % sections
        start_idx = 0
        for split_idx in range(sections):
            split_size = (
                min_split_size + 1
                if (split_idx < num_splits_one_extra)
                else min_split_size
            )
            s = prims.slice_in_dim(a, start_idx, start_idx + split_size, axis=_dim)
            splits.append(s)
            start_idx = start_idx + split_size

        return tuple(splits)
    # Case 1 -- indices_or_sections is a sequence of integers or a 1D tensor describing the splits
    else:
        indices = indices_or_sections
        if isinstance(indices_or_sections, TensorLike):
            if indices_or_sections.ndim != 1:
                msg = "tensor_split: non-scalar indices_or_sections tensors must have only one dimension, "
                "but received a tensor with {0} dimensions".format(
                    indices_or_sections.ndim
                )
                raise ValueError(msg)

            indices = indices_or_sections.tolist()

        splits = []
        start_idx = 0
        for x in indices:
            splits.append(prims.slice_in_dim(a, start_idx, x, axis=_dim))
            start_idx = x
        splits.append(prims.slice_in_dim(a, start_idx, a.shape[_dim], axis=_dim))
        return tuple(splits)


def transpose(a: TensorLikeType, dim0: int, dim1: int) -> TensorLikeType:
    _dim0, _dim1 = utils.canonicalize_dims(a.ndim, (dim0, dim1))  # type: ignore[misc]

    if a.ndim <= 1:
        return prims.view_of(a)

    _permutation = list(range(0, a.ndim))
    _permutation[_dim0] = _dim1
    _permutation[_dim1] = _dim0
    return prims.transpose(a, _permutation)


# Aliases for transpose
swap_axes = transpose


def unsqueeze(a: TensorLikeType, dim: int) -> TensorLikeType:
    # Note that unsqueeze canonicalizes with rank + 1 because it allows
    # a new innermost dimension to be specified
    dim = utils.canonicalize_dim(a.ndim + 1, dim)
    return prims.expand_dims(a, (dim,))


def view(a: TensorLikeType, shape: ShapeType) -> TensorLikeType:
    return _reshape_view_helper(a, shape, allow_copy=False)


@out_wrapper
def empty(
    *shape,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> TensorLikeType:
    shape = utils.extract_shape_from_varargs(shape)
    strides = utils.make_contiguous_strides_for(shape)
    return empty_strided(
        shape, strides, dtype=dtype, device=device, requires_grad=requires_grad
    )


def empty_like(
    a: TensorLikeType,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> TensorLikeType:

    dtype = a.dtype if dtype is None else dtype
    device = a.device if device is None else device

    strides: Tuple[int, ...]
    if a.numel() == 0:
        strides = a.stride()
    else:
        strides = utils.compute_elementwise_output_strides(a)

    return empty_strided(
        a.shape, strides, dtype=dtype, device=device, requires_grad=requires_grad
    )


# NOTE: for convenience, shape can be a tuple of ints or a tuple containing a tuple of ints
def empty_strided(
    shape: Union[ShapeType, Tuple[ShapeType]],
    strides: StrideType,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> TensorLikeType:

    shape = utils.extract_shape_from_varargs(shape)
    dtype = torch.get_default_dtype() if dtype is None else dtype
    device = torch.device("cpu") if device is None else device

    return prims.empty_strided(
        shape, strides, dtype=dtype, device=device, requires_grad=requires_grad
    )


@out_wrapper
def full(
    shape: ShapeType,
    fill_value: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    e = empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    return fill(e, fill_value)


def full_like(
    a: TensorLikeType,
    fill_value: NumberType,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> TensorLikeType:
    e = empty_like(a, dtype=dtype, device=device, requires_grad=requires_grad)
    return fill(e, fill_value)


ones = partial(full, fill_value=True)

ones_like = partial(full_like, fill_value=True)

zeros = partial(full, fill_value=False)

zeros_like = partial(full_like, fill_value=False)


def uniform(
    shape: ShapeType,
    low: Union[bool, int, float] = 0.0,
    high: Union[bool, int, float] = 1.0,
    *,
    dtype: torch.dtype,
    device: DeviceLikeType,
) -> TensorLikeType:
    utils.validate_shape(shape)

    assert isinstance(low, (bool, int, float))
    assert isinstance(high, (bool, int, float))
    low = float(low)
    high = float(high)

    assert isinstance(dtype, torch.dtype)
    device = utils.canonicalize_device(device)

    return prims.uniform(shape, low=low, high=high, dtype=dtype, device=device)


# TODO: add OpInfo for torch.equal and refs.equal
def equal(a: TensorLikeType, b: TensorLikeType) -> bool:
    utils.check_same_device(a, b, allow_cpu_scalar_tensors=False)
    utils.check_same_dtype(a, b)

    # Shape check
    if a.ndim != b.ndim:
        return False

    for x, y in zip(a.shape, b.shape):
        if x != y:
            return False

    # Short-circuits if there are no elements to validate
    if a.numel() == 0:
        return True

    return item(all(eq(a, b)))  # type: ignore[return-value]


# populate the decomp table
import torch._refs.nn.functional
import torch._refs.special
