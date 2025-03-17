//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_APPLY_CV_H
#define _LIBCUDACXX___TYPE_TRAITS_APPLY_CV_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_volatile.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp,
          class _Up,
          bool = is_const<__libcpp_remove_reference_t<_Tp>>::value,
          bool = is_volatile<__libcpp_remove_reference_t<_Tp>>::value>
struct __apply_cv
{
  typedef _LIBCUDACXX_NODEBUG_TYPE _Up type;
};

template <class _Tp, class _Up>
struct __apply_cv<_Tp, _Up, true, false>
{
  typedef _LIBCUDACXX_NODEBUG_TYPE const _Up type;
};

template <class _Tp, class _Up>
struct __apply_cv<_Tp, _Up, false, true>
{
  typedef volatile _Up type;
};

template <class _Tp, class _Up>
struct __apply_cv<_Tp, _Up, true, true>
{
  typedef const volatile _Up type;
};

template <class _Tp, class _Up>
struct __apply_cv<_Tp&, _Up, false, false>
{
  typedef _Up& type;
};

template <class _Tp, class _Up>
struct __apply_cv<_Tp&, _Up, true, false>
{
  typedef const _Up& type;
};

template <class _Tp, class _Up>
struct __apply_cv<_Tp&, _Up, false, true>
{
  typedef volatile _Up& type;
};

template <class _Tp, class _Up>
struct __apply_cv<_Tp&, _Up, true, true>
{
  typedef const volatile _Up& type;
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_APPLY_CV_H
