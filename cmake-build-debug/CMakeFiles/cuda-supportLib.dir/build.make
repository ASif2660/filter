# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

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
CMAKE_COMMAND = /home/asif/Downloads/clion-2018.3.4/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/asif/Downloads/clion-2018.3.4/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/asif/CLionProjects/filter

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/asif/CLionProjects/filter/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/cuda-supportLib.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cuda-supportLib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cuda-supportLib.dir/flags.make

CMakeFiles/cuda-supportLib.dir/cuda-support/src/filter.cpp.o: CMakeFiles/cuda-supportLib.dir/flags.make
CMakeFiles/cuda-supportLib.dir/cuda-support/src/filter.cpp.o: ../cuda-support/src/filter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/asif/CLionProjects/filter/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cuda-supportLib.dir/cuda-support/src/filter.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cuda-supportLib.dir/cuda-support/src/filter.cpp.o -c /home/asif/CLionProjects/filter/cuda-support/src/filter.cpp

CMakeFiles/cuda-supportLib.dir/cuda-support/src/filter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cuda-supportLib.dir/cuda-support/src/filter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/asif/CLionProjects/filter/cuda-support/src/filter.cpp > CMakeFiles/cuda-supportLib.dir/cuda-support/src/filter.cpp.i

CMakeFiles/cuda-supportLib.dir/cuda-support/src/filter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cuda-supportLib.dir/cuda-support/src/filter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/asif/CLionProjects/filter/cuda-support/src/filter.cpp -o CMakeFiles/cuda-supportLib.dir/cuda-support/src/filter.cpp.s

# Object files for target cuda-supportLib
cuda__supportLib_OBJECTS = \
"CMakeFiles/cuda-supportLib.dir/cuda-support/src/filter.cpp.o"

# External object files for target cuda-supportLib
cuda__supportLib_EXTERNAL_OBJECTS =

lib/libcuda-supportLib.a: CMakeFiles/cuda-supportLib.dir/cuda-support/src/filter.cpp.o
lib/libcuda-supportLib.a: CMakeFiles/cuda-supportLib.dir/build.make
lib/libcuda-supportLib.a: CMakeFiles/cuda-supportLib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/asif/CLionProjects/filter/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library lib/libcuda-supportLib.a"
	$(CMAKE_COMMAND) -P CMakeFiles/cuda-supportLib.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuda-supportLib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cuda-supportLib.dir/build: lib/libcuda-supportLib.a

.PHONY : CMakeFiles/cuda-supportLib.dir/build

CMakeFiles/cuda-supportLib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cuda-supportLib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cuda-supportLib.dir/clean

CMakeFiles/cuda-supportLib.dir/depend:
	cd /home/asif/CLionProjects/filter/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/asif/CLionProjects/filter /home/asif/CLionProjects/filter /home/asif/CLionProjects/filter/cmake-build-debug /home/asif/CLionProjects/filter/cmake-build-debug /home/asif/CLionProjects/filter/cmake-build-debug/CMakeFiles/cuda-supportLib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cuda-supportLib.dir/depend
