#ifndef __BM1880V2_TPU_CFG__
#define __BM1880V2_TPU_CFG__

#define BM1880V2_VER                                 18802
#define BM1880V2_HW_NPU_SHIFT                        5
#define BM1880V2_HW_EU_SHIFT                         4
#define BM1880V2_HW_LMEM_SHIFT                       15
#define BM1880V2_HW_LMEM_BANKS                       8
#define BM1880V2_HW_LMEM_BANK_SIZE                   0x1000
#define BM1880V2_HW_NODE_CHIP_SHIFT                  0
#define BM1880V2_HW_NPU_NUM                          (1 << BM1880V2_HW_NPU_SHIFT)
#define BM1880V2_HW_EU_NUM                           (1 << BM1880V2_HW_EU_SHIFT)
#define BM1880V2_HW_LMEM_SIZE                        (1 << BM1880V2_HW_LMEM_SHIFT)
#define BM1880V2_HW_NODE_CHIP_NUM                    (1 << BM1880V2_HW_NODE_CHIP_SHIFT)

#if (BM1880V2_HW_LMEM_SIZE != (BM1880V2_HW_LMEM_BANK_SIZE * BM1880V2_HW_LMEM_BANKS))
#error "Set wrong TPU configuraiton."
#endif

#define BM1880V2_GLOBAL_MEM_START_ADDR               0x100000000
#define BM1880V2_GLOBAL_MEM_SIZE                     0x100000000

#define BM1880V2_GLOBAL_TIU_CMDBUF_ADDR              0x00000000
#define BM1880V2_GLOBAL_TDMA_CMDBUF_ADDR             0x01400000
#define BM1880V2_GLOBAL_TIU_CMDBUF_RESERVED_SIZE     0x01400000
#define BM1880V2_GLOBAL_TDMA_CMDBUF_RESERVED_SIZE    0x01400000
#define BM1880V2_GLOBAL_POOL_RESERVED_SIZE           (BM1880V2_GLOBAL_MEM_SIZE - BM1880V2_GLOBAL_TIU_CMDBUF_RESERVED_SIZE - BM1880V2_GLOBAL_TDMA_CMDBUF_RESERVED_SIZE)

#define BM1880V2_UART_CTLR_BASE_ADDR                 0x04140000

#define BM1880V2_TDMA_ENGINE_BASE_ADDR               0x0C100000
#define BM1880V2_TDMA_ENGINE_END_ADDR                (BM1880V2_TDMA_ENGINE_BASE_ADDR + 0x1000)

#define BM1880V2_TIU_ENGINE_BASE_ADDR                0x0C101000 //"NPS Register" in memory map?
#define BM1880V2_TIU_ENGINE_END_ADDR                 (BM1880V2_TIU_ENGINE_BASE_ADDR + 0x1000)

#endif
