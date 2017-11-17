/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Created by: Hang Zhang
 * ECE Department, Rutgers University
 * Email: zhang.hang@rutgers.edu
 * Copyright (c) 2017
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree 
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */
#include "thc_encoding.h"
#include "common.h"

#include "generic/device_tensor.h"
#include "THC/THCGenerateFloatType.h"

#include "generic/device_tensor.h"
#include "THC/THCGenerateDoubleType.h"

#ifdef __cplusplus
extern "C" {
#endif

// float
#include "generic/encoding_utils.c"
#include "THC/THCGenerateFloatType.h"

#include "generic/encoding_kernel.c"
#include "THC/THCGenerateFloatType.h"

#include "generic/syncbn_kernel.c"
#include "THC/THCGenerateFloatType.h"

#include "generic/pooling_kernel.c"
#include "THC/THCGenerateFloatType.h"

// double
#include "generic/encoding_utils.c"
#include "THC/THCGenerateDoubleType.h"

#include "generic/encoding_kernel.c"
#include "THC/THCGenerateDoubleType.h"

#include "generic/syncbn_kernel.c"
#include "THC/THCGenerateDoubleType.h"

#include "generic/pooling_kernel.c"
#include "THC/THCGenerateDoubleType.h"

#ifdef __cplusplus
}
#endif
