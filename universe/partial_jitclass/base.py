"""Partial numba.experimental.jitclass implementation"""

from typing import Callable

import numba
import numpy as np
from numba.core import config, errors

# implicitly import everything needed from jitclass.base
from numba.experimental.jitclass.base import (
    ClassBuilder,
    ConstructorTemplate,
    JitClassType,
    OrderedDict,
    Sequence,
    _add_linking_libs,
    _drop_ignored_attrs,
    _fix_up_private_attr,
    _validate_spec,
    as_numba_type,
    cgutils,
    cpu_target,
    default_manager,
    imp_dtor,
    imputils,
    inspect,
    models,
    njit,
    pt,
    pytypes,
    types,
)
from numba.types import string


def convert_to_numba(input_type):
    """Convert input to numba type"""
    try:
        return as_numba_type(input_type)
    except errors.TypingError:
        return (
            input_type
            if isinstance(input_type, numba.types.Type)
            else (
                numba.void
                if input_type == np.void0
                else (
                    string
                    if input_type == np.str0
                    else (
                        input_type.class_type.instance_type
                        if hasattr(input_type, "class_type")
                        else numba.from_dtype(input_type)
                    )
                )
            )
        )


def njit_spec(*args, **kwargs) -> pytypes.FunctionType:
    """Decorator factory to specify numba.njit arguments for a numba.experimental.jitclass method

    Returns:
        pytypes.FunctionType: Decorator for method with specification
    """

    def decorator(method: pytypes.FunctionType) -> pytypes.FunctionType:
        """Decorator for a method which stores the njit specification within the function object

        Args:
            method (pytypes.FunctionType): Method to be decorated

        Returns:
            pytypes.FunctionType: Method now with njit_args, njit_kwargs, and njit attributes
        """

        # store njit spec within method
        method.njit_args = args
        method.njit_kwargs = kwargs
        method.njit = True

        return method

    return decorator


def py_func(method):
    """Decorator for a non-compiled method only accessable to the interpreter

    Args:
        method (pytypes.FunctionType): Method to be decorated

    Returns:
        pytypes.FunctionType: Method now with njit attribute
    """
    method.njit = False

    return method


def apply_njit(
    class_type, method: pytypes.FunctionType, instance_method: bool = False
) -> Callable:
    """Apply njit specification to method and return dispatched njit callable

    Args:
        method (pytypes.FunctionType): Function with njit attribute

    Returns:
        Callable: Dispatched njit callable
    """

    # If a method does not have njit_spec or py_func, default to inferred compilation
    if not hasattr(method, "njit"):
        return njit(method)

    # Do not compile py_func functions
    if not method.njit:
        return method

    # Manually force the first argument of instance methods to be the instance type (self argument)
    if len(method.njit_args) > 0 and instance_method:
        method.njit_args[0]._args = (class_type.instance_type,) + method.njit_args[
            0
        ]._args

    return njit(
        *(method.njit_args if hasattr(method, "njit_args") else ()),
        **(method.njit_kwargs if hasattr(method, "njit_kwargs") else {}),
    )(method)


class PartialJitClassType(JitClassType):
    """
    The type of any partial jitclass.
    """

    instance_type_class = None

    def __new__(cls, name, bases, dct):
        if len(bases) != 1:
            raise TypeError("must have exactly one base class")
        [base] = bases
        if isinstance(base, JitClassType):
            raise TypeError("cannot subclass from a jitclass")
        assert "class_type" in dct, 'missing "class_type" attr'
        return type.__new__(cls, name, bases, dct)


class PartialClassType(numba.types.ClassType):
    """
    The type of the partial jitted class (not instance). When the type of a class
    is called, its constructor is invoked.
    """

    def __init__(
        self,
        class_def,
        ctor_template_cls,
        struct,
        jit_methods,
        jit_props,
        jit_static_methods,
    ):
        self.class_name = class_def.__name__
        self.class_doc = class_def.__doc__
        self._ctor_template_class = ctor_template_cls
        self.jit_methods = jit_methods
        self.jit_props = jit_props
        self.jit_static_methods = jit_static_methods
        self.struct = struct
        fielddesc = ",".join(f"{k}:{v}" for k, v in struct.items())
        name = f"{self.name_prefix}.{self.class_name}#{id(self):x}<{fielddesc}>"
        super(numba.types.ClassType, self).__init__(name)

    @property
    def methods(self):
        return {k: getattr(v, "py_func", v) for k, v in self.jit_methods.items()}

    @property
    def static_methods(self):
        return {k: getattr(v, "py_func", v) for k, v in self.jit_static_methods.items()}


default_manager.register(PartialClassType, models.OpaqueModel)


class PartialClassBuilder(ClassBuilder):
    """
    A partial jitclass builder for a mutable partial jitclass. This will register
    typing and implementation hooks to the given typing and target contexts.
    """


@PartialClassBuilder.class_impl_registry.lower(
    PartialClassType, types.VarArg(types.Any)
)
def ctor_impl(context, builder, sig, args):
    """
    Generic constructor (__new__) for partial jitclasses.
    """
    # Allocate the instance
    inst_typ = sig.return_type
    alloc_type = context.get_data_type(inst_typ.get_data_type())
    alloc_size = context.get_abi_sizeof(alloc_type)

    meminfo = context.nrt.meminfo_alloc_dtor(
        builder,
        context.get_constant(types.uintp, alloc_size),
        imp_dtor(context, builder.module, inst_typ),
    )
    data_pointer = context.nrt.meminfo_data(builder, meminfo)
    data_pointer = builder.bitcast(data_pointer, alloc_type.as_pointer())

    # Nullify all data
    builder.store(cgutils.get_null_value(alloc_type), data_pointer)

    inst_struct = context.make_helper(builder, inst_typ)
    inst_struct.meminfo = meminfo
    inst_struct.data = data_pointer

    # Call the jitted __init__
    # TODO: extract the following into a common util
    init_sig = (sig.return_type,) + sig.args

    init = inst_typ.jit_methods["__init__"]
    disp_type = types.Dispatcher(init)
    call = context.get_function(disp_type, types.void(*init_sig))
    _add_linking_libs(context, call)
    realargs = [inst_struct._getvalue()] + list(args)
    call(builder, realargs)

    # Prepare return value
    ret = inst_struct._getvalue()

    return imputils.impl_ret_new_ref(context, builder, inst_typ, ret)


def register_class_type(cls, spec, class_ctor, builder):
    """
    Internal function to create a partial jitclass.

    Args
    ----
    cls: the original class object (used as the prototype)
    spec: the structural specification contains the field types.
    class_ctor: the numba type to represent the jitclass
    builder: the internal jitclass builder
    """

    # Normalize spec
    if spec is None:
        spec = OrderedDict()
    elif isinstance(spec, Sequence):
        spec = OrderedDict(spec)

    # Extend spec with class annotations.
    for attr, py_type in pt.get_type_hints(cls).items():
        if attr not in spec:
            spec[attr] = convert_to_numba(py_type)

    _validate_spec(spec)

    # Fix up private attribute names
    spec = _fix_up_private_attr(cls.__name__, spec)

    # Copy methods from base classes
    clsdct = {}
    for basecls in reversed(inspect.getmro(cls)):
        clsdct |= basecls.__dict__

    methods, props, static_methods, others = {}, {}, {}, {}
    for k, v in clsdct.items():
        if isinstance(v, pytypes.FunctionType):
            methods[k] = v
        elif isinstance(v, property):
            props[k] = v
        elif isinstance(v, staticmethod):
            static_methods[k] = v
        else:
            others[k] = v

    if shadowed := (set(methods) | set(props) | set(static_methods)) & set(spec):
        raise NameError(f"name shadowing: {', '.join(shadowed)}")

    docstring = others.pop("__doc__", "")
    _drop_ignored_attrs(others)
    if others:
        members = ", ".join(others.keys())
        raise TypeError(f"class members are not yet supported: {members}")

    for k, v in props.items():
        if v.fdel is not None:
            raise TypeError(f"deleter is not supported: {k}")

    # TODO: other jit methods
    jit_props = {}
    # for k, v in props.items():
    #     dct = {}
    #     if v.fget:
    #         dct["get"] = apply_njit(cls, v.fget)
    #     if v.fset:
    #         dct["set"] = apply_njit(cls, v.fset)
    #     jit_props[k] = dct

    jit_static_methods = {
        # k: apply_njit(cls, v.__func__) for k, v in static_methods.items()
    }

    # Instantiate class type without compiled methods
    class_type = class_ctor(
        cls, ConstructorTemplate, spec, methods, jit_props, jit_static_methods
    )

    jit_class_dct = {
        "class_type": class_type,
        "__doc__": docstring,
    } | jit_static_methods

    cls = PartialJitClassType(cls.__name__, (cls,), jit_class_dct)
    # Register resolution of the class object
    typingctx = cpu_target.typing_context
    typingctx.insert_global(cls, class_type)

    # Register class
    targetctx = cpu_target.target_context
    builder(class_type, typingctx, targetctx).register()
    as_numba_type.register(cls, class_type.instance_type)

    # Compile after class is registered
    class_type.jit_methods = {
        k: apply_njit(class_type, v, instance_method=True)
        for k, v in class_type.methods.items()
    }
    # Set initialized after __init__ is compiled
    cls._set_init()

    return cls


def partial_jitclass(cls_or_spec=None, spec=None):
    """
    A function for creating a partial jitclass.

    Partial jitclasses differ from normal jitclasses as they allow the user to
    specify the njit arguments of methods via the @njit_spec decorator.
    Additionally, partial jitclasses allow the user to specify non-compiled
    methods which can only be accessed by the interpreter, via the @py_func
    decorator.

    Can be used as a decorator or function.
    """

    if cls_or_spec is not None and spec is None and not isinstance(cls_or_spec, type):
        # Used like
        # @partial_jitclass([("x", intp)])
        # class Foo:
        #     ...
        spec = cls_or_spec
        cls_or_spec = None

    def wrap(cls):
        # pylint: disable=no-member
        if config.DISABLE_JIT:
            return cls
        # pylint: enable=no-member
        cls_jitted = register_class_type(
            cls, spec, PartialClassType, PartialClassBuilder
        )

        # Preserve the module name of the original class
        cls_jitted.__module__ = cls.__module__

        return cls_jitted

    return wrap if cls_or_spec is None else wrap(cls_or_spec)
