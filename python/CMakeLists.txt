# Copyright 2011 Free Software Foundation, Inc.
#
# This file was generated by gr_modtool, a tool from the GNU Radio framework
# This file is a part of gr-dvbs2rx
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Include python install macros
########################################################################
include(GrPython)
if(NOT PYTHONINTERP_FOUND)
    return()
endif()

add_subdirectory(bindings)

########################################################################
# Install python sources
########################################################################
GR_PYTHON_INSTALL(
    FILES
    __init__.py
	defs.py
	params.py
    DESTINATION ${GR_PYTHON_DIR}/dvbs2rx
)

########################################################################
# Handle the unit tests
########################################################################
include(GrTest)

set(GR_TEST_TARGET_DEPS gnuradio-dvbs2rx)
set(GR_TEST_ENVIRONS PYTHONPATH=${CMAKE_BINARY_DIR})
GR_ADD_TEST(qa_bbdeheader_bb ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_bbdeheader_bb.py)
GR_ADD_TEST(qa_params ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_params.py)
GR_ADD_TEST(qa_plsync_cc ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_plsync_cc.py)
GR_ADD_TEST(qa_symbol_sync_cc ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_symbol_sync_cc.py)
