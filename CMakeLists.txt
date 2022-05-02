add_library(app_lib INTERFACE)
target_sources(app_lib INTERFACE
lapi.cpp
lcode.cpp
ldebug.cpp
ldo.cpp
ldump.cpp
lfunc.cpp
lgc.cpp
llex.cpp
lmem.cpp
lobject.cpp
lopcodes.cpp
lparser.cpp
lstate.cpp
lstring.cpp
ltable.cpp
ltm.cpp

lundump.cpp
lvm.cpp
lzio.cpp
lctype.cpp
lauxlib.cpp
lbaselib.cpp
lcorolib.cpp
ldblib.cpp
ltablib.cpp
lstrlib.cpp
lpico8lib.cpp
linit.cpp

)
target_link_libraries(app_lib INTERFACE pico_stdlib)
