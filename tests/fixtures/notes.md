# Lab notes

Short intro paragraph about the lab board.

## UART interrupts

The USART peripheral raises an RXNE interrupt when a byte is received. The ISR should read the data register quickly to avoid overrun errors. Measure latency with a GPIO toggle and a logic analyzer.

```c
void USART2_IRQHandler(void) { /* ## not a heading */ }
```

## Clock tree

The system clock can come from HSI, HSE or the PLL. The PLL multiplies the HSE crystal frequency to reach 168 MHz.
