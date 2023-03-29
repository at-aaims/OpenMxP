#ifndef LOG_HPP
#define LOG_HPP

// remove the following will compile away debug message, but keep info
// or make it conditional by: #ifdef DEBUG
#if 0
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include "spdlog/cfg/env.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#endif

//extern std::shared_ptr<spdlog::logger> LOG;
  
extern int grank, gsize, glog;

// #define INFO(...) {LOG->info(__VA_ARGS__); }

// #define WARN(...) {LOG->warn(__VA_ARGS__); }

#if 0
#ifdef NDEBUG
#define TRACE(...) (void)0
#else
#define TRACE(...) SPDLOG_LOGGER_CALL(spdlog::default_logger_raw(), spdlog::level::trace, __VA_ARGS__) 
#endif
#endif

void PrintMsg(const char *fmt...) {
    if (grank == 0) {
//      LOG->info(fmt);
    }
}

// #define PrintLogMsg( ... ) { if ( grank == 0  &&  glog == 1 ) LOG->info( __VA_ARGS__ ); }

#define OUTPUT if ( grank == 0  &&  glog == 1 ) 

inline
void PrintLogMsg( const char * fmt, ... )
{
   if ( grank == 0  &&  glog == 1 )
   {
      va_list argPtr;
      va_start( argPtr, fmt );
      vfprintf( stdout, fmt, argPtr );
      va_end( argPtr );
      fflush( stdout );
   }
}

#endif
