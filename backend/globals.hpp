#pragma once

/** 
 *  Copyright 2023, 2024 Philipp Andelfinger, Justin Kreikemeyer
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 *  and associated documentation files (the “Software”), to deal in the Software without
 *  restriction, including without limitation the rights to use, copy, modify, merge, publish,
 *  distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
 *  Software is furnished to do so, subject to the following conditions:
 *   
 *    The above copyright notice and this permission notice shall be included in all copies or
 *    substantial portions of the Software.
 *    
 *    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 *    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 *    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 *    ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 *    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *    SOFTWARE.
 */

uint64_t global_branch_id;
uint32_t branch_level;
const uint64_t initial_global_branch_id = 11061421359639307453UL; // arbitrary 64 bit

#ifndef DGO_FORK_LIMIT
#define DGO_FORK_LIMIT 0
#endif

#ifndef DGO_MIN_COND_VARIANCE
#define DGO_MIN_COND_VARIANCE 0
//#define DGO_MIN_COND_VARIANCE 1e-10
#endif
