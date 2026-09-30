"""Regenerates the synthetic test manuals (needs reportlab):

  um9999.pdf    - board-manual style, running header/footer, TOC page, no outline
  rm9999.pdf    - reference-manual style with a PDF outline (bookmarks), register
                  descriptions with bit rulers, front/back matter to be skipped
"""
import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

HERE = Path(__file__).parent
W, H = A4


def board_manual() -> None:
    c = canvas.Canvas(str(HERE / "um9999.pdf"), pagesize=A4)
    n = 6
    body = {
        1: [("TOC", None)],
        2: [("H", "1 Features"), ("P", "The demo kit features an ARM Cortex-M4 microcontroller with 1 Mbyte of Flash memory and 192 Kbytes of RAM in an LQFP100 package. " * 3),
            ("H", "2 Hardware layout and configuration"), ("P", "The kit is designed around the microcontroller. The hardware block diagram illustrates the connection between the MCU and peripherals such as the debugger, LEDs and push-buttons. " * 4)],
        3: [("H", "2.1 Power supply"), ("P", "The power supply is provided either by the host PC through the USB cable, or by an external 5 V power supply. The on-board regulator generates 3 V for the microcontroller. " * 5),
            ("P", "3 V power supply rail is available on the extension header.")],
        4: [("H", "2.2 LEDs"), ("P", "LD3 orange, LD4 green, LD5 red and LD6 blue are user LEDs connected to the I/O pins PD13, PD12, PD14 and PD15 respectively. " * 4),
            ("H", "2.3 Push-buttons"), ("P", "B1 USER: the user push-button is connected to the I/O PA0. It can also be used as a wake-up button. B2 RESET: the push-button is connected to NRST and is used to reset the microcontroller. " * 3)],
        5: [("H", "2.4 USB OTG supported"), ("P", "The microcontroller supports USB OTG full-speed communication via a micro-AB connector. A green LED indicates when VBUS is present. " * 6)],
        6: [("H", "3 Revision history"), ("P", "Initial release of the document.")],
    }
    for p in range(1, n + 1):
        c.setFont("Helvetica", 8)
        c.drawString(50, H - 30, "UM9999 Hardware layout")
        c.drawString(50, 20, "UM9999 Rev 3")
        c.drawString(W - 80, 20, f"{p}/{n}")
        y = H - 100
        c.setFont("Helvetica", 10)
        for kind, txt in body[p]:
            if kind == "TOC":
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y, "Contents")
                y -= 20
                c.setFont("Helvetica", 10)
                for t, pg in [("1 Features", 2), ("2 Hardware layout and configuration", 2), ("2.1 Power supply", 3), ("2.2 LEDs", 4), ("2.3 Push-buttons", 4)]:
                    c.drawString(50, y, f"{t} " + "." * 40 + f" {pg}")
                    y -= 14
                continue
            if kind == "H":
                y -= 6
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y, txt)
                y -= 18
                c.setFont("Helvetica", 10)
                continue
            for line in textwrap.wrap(txt, 95):
                c.drawString(50, y, line)
                y -= 13
            y -= 8
        c.showPage()
    c.save()


def reference_manual() -> None:
    c = canvas.Canvas(str(HERE / "rm9999.pdf"), pagesize=A4)
    pages = [
        [("L1", "Contents"), ("P", "1 GPIO . . . . . . . . . . . . . . . 2"), ("P", "2 USART . . . . . . . . . . . . . . 3")],
        [("L1", "1 General-purpose I/Os (GPIO)"), ("P", "Each GPIO port has four 32-bit configuration registers. " * 3),
         ("L2", "1.1 GPIO registers"), ("P", "The GPIO registers can be accessed by byte, half-word or word."),
         ("L3", "1.1.1 GPIO port mode register (GPIOx_MODER)"), ("P", "Address offset: 0x00"),
         ("P", "Reset value: 0xA800 0000 for port A"),
         ("R", "31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16"),
         ("R", "rw rw rw rw rw rw rw rw rw rw rw rw rw rw rw rw"),
         ("P", "Bits 2y:2y+1 MODERy[1:0]: Port x configuration bits. 00: Input, 01: General purpose output mode, 10: Alternate function mode, 11: Analog mode.")],
        [("L1", "2 Universal synchronous asynchronous receiver transmitter (USART)"),
         ("P", "The USART offers a flexible means of full-duplex data exchange with external equipment. " * 3),
         ("L2", "2.1 Status register (USART_SR)"), ("P", "Address offset: 0x00"),
         ("P", "Bit 7 TXE: Transmit data register empty. Bit 5 RXNE: Read data register not empty. " * 2)],
        [("L1", "Revision history"), ("P", "Rev 1: initial release. Changed OSPEEDR bits description. " * 3)],
    ]
    n = len(pages)
    for pno, items in enumerate(pages, start=1):
        c.setFont("Helvetica", 8)
        c.drawString(50, H - 45, "RM9999 Reference manual")
        c.drawString(250, H - 800, f"RM9999 Rev 1 {pno}/{n}")
        y = H - 100
        for kind, txt in items:
            if kind in ("L1", "L2", "L3"):
                key = f"k{pno}_{txt[:12]}"
                c.bookmarkPage(key, fit="XYZ", left=0, top=y + 14)
                c.addOutlineEntry(txt, key, level=int(kind[1]) - 1)
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y, txt)
                y -= 20
                continue
            c.setFont("Helvetica", 10)
            for line in textwrap.wrap(txt, 95):
                c.drawString(50, y, line)
                y -= 13
            y -= 6
        c.showPage()
    c.save()


if __name__ == "__main__":
    board_manual()
    reference_manual()
