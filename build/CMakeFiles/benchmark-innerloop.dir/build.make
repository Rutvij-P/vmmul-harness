# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /global/homes/j/jivtur/vmmul-harness

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /global/homes/j/jivtur/vmmul-harness/build

# Include any dependencies generated for this target.
include CMakeFiles/benchmark-innerloop.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/benchmark-innerloop.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/benchmark-innerloop.dir/flags.make

CMakeFiles/benchmark-innerloop.dir/dgemv-innerloop.cpp.o: CMakeFiles/benchmark-innerloop.dir/flags.make
CMakeFiles/benchmark-innerloop.dir/dgemv-innerloop.cpp.o: ../dgemv-innerloop.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/homes/j/jivtur/vmmul-harness/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/benchmark-innerloop.dir/dgemv-innerloop.cpp.o"
	/opt/cray/pe/craype/2.7.10/bin/CC  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/benchmark-innerloop.dir/dgemv-innerloop.cpp.o -c /global/homes/j/jivtur/vmmul-harness/dgemv-innerloop.cpp

CMakeFiles/benchmark-innerloop.dir/dgemv-innerloop.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchmark-innerloop.dir/dgemv-innerloop.cpp.i"
	/opt/cray/pe/craype/2.7.10/bin/CC $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /global/homes/j/jivtur/vmmul-harness/dgemv-innerloop.cpp > CMakeFiles/benchmark-innerloop.dir/dgemv-innerloop.cpp.i

CMakeFiles/benchmark-innerloop.dir/dgemv-innerloop.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchmark-innerloop.dir/dgemv-innerloop.cpp.s"
	/opt/cray/pe/craype/2.7.10/bin/CC $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /global/homes/j/jivtur/vmmul-harness/dgemv-innerloop.cpp -o CMakeFiles/benchmark-innerloop.dir/dgemv-innerloop.cpp.s

# Object files for target benchmark-innerloop
benchmark__innerloop_OBJECTS = \
"CMakeFiles/benchmark-innerloop.dir/dgemv-innerloop.cpp.o"

# External object files for target benchmark-innerloop
benchmark__innerloop_EXTERNAL_OBJECTS = \
"/global/homes/j/jivtur/vmmul-harness/build/CMakeFiles/benchmark.dir/benchmark.cpp.o"

benchmark-innerloop: CMakeFiles/benchmark-innerloop.dir/dgemv-innerloop.cpp.o
benchmark-innerloop: CMakeFiles/benchmark.dir/benchmark.cpp.o
benchmark-innerloop: CMakeFiles/benchmark-innerloop.dir/build.make
benchmark-innerloop: CMakeFiles/benchmark-innerloop.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/global/homes/j/jivtur/vmmul-harness/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable benchmark-innerloop"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/benchmark-innerloop.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/benchmark-innerloop.dir/build: benchmark-innerloop

.PHONY : CMakeFiles/benchmark-innerloop.dir/build

CMakeFiles/benchmark-innerloop.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/benchmark-innerloop.dir/cmake_clean.cmake
.PHONY : CMakeFiles/benchmark-innerloop.dir/clean

CMakeFiles/benchmark-innerloop.dir/depend:
	cd /global/homes/j/jivtur/vmmul-harness/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /global/homes/j/jivtur/vmmul-harness /global/homes/j/jivtur/vmmul-harness /global/homes/j/jivtur/vmmul-harness/build /global/homes/j/jivtur/vmmul-harness/build /global/homes/j/jivtur/vmmul-harness/build/CMakeFiles/benchmark-innerloop.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/benchmark-innerloop.dir/depend
