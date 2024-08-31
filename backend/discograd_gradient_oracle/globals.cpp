#include <cstdint>

uint64_t global_branch_id;                                                                                            
uint32_t branch_level;

#if DGO_FORK_LIMIT > 0
const uint64_t initial_global_branch_id = 11061421359639307453UL; // arbitrary 64 bit
#endif
