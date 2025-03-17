//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_UNSIGNED_INTEGER_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_UNSIGNED_INTEGER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __libcpp_is_unsigned_integer : public false_type
{};
template <>
struct __libcpp_is_unsigned_integer<unsigned char> : public true_type
{};
template <>
struct __libcpp_is_unsigned_integer<unsigned short> : public true_type
{};
template <>
struct __libcpp_is_unsigned_integer<unsigned int> : public true_type
{};
template <>
struct __libcpp_is_unsigned_integer<unsigned long> : public true_type
{};
template <>
struct __libcpp_is_unsigned_integer<unsigned long long> : public true_type
{};
#ifndef _LIBCUDACXX_HAS_NO_INT128
template <>
struct __libcpp_is_unsigned_integer<__uint128_t> : public true_type
{};
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_UNSIGNED_INTEGER_H
