#ifndef DEBUG_H
#define DEBUG_H

void tensor_panic(const char *file, int line, const char *expr);
void tensor_printf(const char *file, int line, const char *fmt, ...);
//
/*
#ifdef assert
#undef assert
#endif

#ifndef NDEBUG
#define debug(...) \
    tensor_printf(__FILE__, __LINE__, __VA_ARGS__)
#define assert(e) \
    do { if (!(e)) tensor_panic(__FILE__, __LINE__, #e); } while(0)
    // ((void)(e) ? 0 : tensor_panic(__FILE__, __LINE__, #e))
#else
#define tensor_debug(...)
#define assert(e)
#endif // !NDEBUG
*/

#endif // !DEBUG_H
