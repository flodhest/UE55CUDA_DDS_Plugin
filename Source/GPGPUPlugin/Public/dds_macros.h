#pragma once

class dds_macros {
public:

    dds_macros() {

#ifndef DDSRT_WITH_FREERTOS
#define DDSRT_WITH_FREERTOS 0
#endif

#ifndef __MINGW32__
#define __MINGW32__ 0
#endif

#ifndef __GNUC__
#define __GNUC__ 0
#endif

#ifndef __clang__
#define __clang__ 0
#endif

#ifndef __GNUC_MINOR__
#define __GNUC_MINOR__ 0
#endif

#ifndef __GNUC_PATCHLEVEL__
#define __GNUC_PATCHLEVEL__ 0
#endif

#ifndef __clang_major__
#define __clang_major__ 0
#endif

#ifndef __clang_minor__
#define __clang_minor__ 0
#endif

#ifndef __clang_patchlevel__
#define __clang_patchlevel__ 0
#endif
    }
    ~dds_macros() = default;
};
