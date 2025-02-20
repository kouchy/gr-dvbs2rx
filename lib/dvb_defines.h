/* -*- c++ -*- */
/*
 * Copyright 2018 Ron Economos.
 *
 * This file is part of gr-dvbs2rx.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_DVBS2RX_DVB_DEFINES_H
#define INCLUDED_DVBS2RX_DVB_DEFINES_H

#define TRUE 1
#define FALSE 0

#define BB_HEADER_LENGTH_BITS 80

// BB HEADER fields
#define TS_GS_TRANSPORT 3
#define TS_GS_GENERIC_PACKETIZED 0
#define TS_GS_GENERIC_CONTINUOUS 1
#define TS_GS_RESERVED 2

#define SIS_MIS_SINGLE 1
#define SIS_MIS_MULTIPLE 0

#define CCM 1
#define ACM 0

#define ISSYI_ACTIVE 1
#define ISSYI_NOT_ACTIVE 0

#define NPD_ACTIVE 1
#define NPD_NOT_ACTIVE 0

#define FRAME_SIZE_NORMAL 64800
#define FRAME_SIZE_MEDIUM 32400
#define FRAME_SIZE_SHORT 16200

// BCH Code
#define BCH_CODE_N8 0
#define BCH_CODE_N10 1
#define BCH_CODE_N12 2
#define BCH_CODE_S12 3
#define BCH_CODE_M12 4

#define LDPC_ENCODE_TABLE_LENGTH (FRAME_SIZE_NORMAL * 10)

#define NORMAL_PUNCTURING 3240
#define MEDIUM_PUNCTURING 1620
#define SHORT_PUNCTURING_SET1 810
#define SHORT_PUNCTURING_SET2 1224

#define VLSNR_OFF 0
#define VLSNR_SET1 1
#define VLSNR_SET2 2

#define EXTRA_PILOT_SYMBOLS_SET1 ((18 * 34) + (3 * 36))
#define EXTRA_PILOT_SYMBOLS_SET2 ((9 * 32) + 36)

#endif /* INCLUDED_DVBS2RX_DVB_DEFINES_H */
