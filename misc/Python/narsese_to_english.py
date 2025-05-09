"""
 * The MIT License
 *
 * Copyright 2020 The OpenNARS authors.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * """

import sys
import re

GREEN = "\x1B[32m"
YELLOW = "\x1B[33m"
CYAN = "\x1B[36m"
MAGENTA = "\x1B[35m"
RED = "\x1B[31m"
RESET = "\x1B[0m"
BOLD = "\x1B[1m"
STATEMENT_OPENER = r""
STATEMENT_CLOSER = r""

def narseseToEnglish_noColors():
    global GREEN, YELLOW, CYAN, MAGENTA, RED, RESET, BOLD
    GREEN = ""
    YELLOW = ""
    CYAN = ""
    MAGENTA = ""
    RED = ""
    RESET = ""
    BOLD = ""
    
narseseToEnglish_noColors()

def narseseToEnglish(line):
    COLOR = GREEN
    line = line.rstrip().replace("(! ", CYAN + "not " + COLOR).replace("#1","it").replace("$1","it").replace("#2","thing").replace("$2","thing")
    if line.startswith("performing ") or line.startswith("done with"):
        COLOR = CYAN
    elif line.startswith("Comment: expected:"):
        COLOR = BOLD + MAGENTA
    elif line.startswith("Comment:") or line.startswith("//"):
        COLOR = MAGENTA
    elif line.startswith("Input:"):
        COLOR = GREEN
    elif line.startswith("Derived:") or line.startswith("Revised:"):
        COLOR = YELLOW
    elif line.startswith("Answer:") or line.startswith("^") or "decision expectation" in line:
        COLOR = BOLD + RED
    if not (" | " in line or " \\1 " in line or " \\2 " in line or " - " in line or " ~ " in line):
        #Ext and Int set
        l = re.sub(r"{([^><:\(\)\*]*)}", MAGENTA+r"" + GREEN + r"\1" + MAGENTA + "" + COLOR, line)
        l = re.sub(r"\[([^><:\(\)\*]*)\]", MAGENTA+r"" + GREEN + r"\1" + MAGENTA + "" + COLOR, l)
        #Image
        l = re.sub(r"\(([^><:]*)\s(/1|\\1|/2|\\1|\\2)\s([^><:]*)\)", YELLOW+r"" + GREEN + r"\1" + YELLOW + r" \2 " + GREEN + r"\3" + YELLOW + "" + COLOR, l)
        #Implication
        l = re.sub(r"<([^:]*)\s=(/|=|\|)>\s([^:]*)>", CYAN+ STATEMENT_OPENER + GREEN + r"\1" + CYAN + r" leads to " + GREEN + r"\3" + CYAN + STATEMENT_CLOSER + COLOR, l)
        #Conjunction
        l = re.sub(r"\(([^:]*)\s&(/|&|\|)\s([^:]*)\)", MAGENTA+r"" + GREEN + r"\1" + MAGENTA + r" and " + GREEN + r"\3" + MAGENTA + "" + COLOR, l)
        #Similarity and inheritance
        l = re.sub(r"<([^><:]*)\s(-->)\s([^><:]*)>", RED+ STATEMENT_OPENER + GREEN + r"\1" + RED + r" is " + GREEN + r"\3" + RED + STATEMENT_CLOSER + COLOR, l)
        l = re.sub(r"<([^><:]*)\s(<->)\s([^><:]*)>", RED+ STATEMENT_OPENER + GREEN + r"\1" + RED + r" resembles " + GREEN + r"\3" + RED + STATEMENT_CLOSER + COLOR, l)
        #Other compound term copulas (not higher order)
        l = re.sub(r"\(([^><:]*)\s(\*|&)\s([^><:]*)\)", YELLOW+r"" + GREEN + r"\1" + YELLOW + r" " + GREEN + r"\3" + YELLOW + "" + COLOR, l)
        return COLOR + l.replace(")","").replace("(","").replace("||", MAGENTA + "or" + COLOR).replace("==>", CYAN + "implies" + COLOR).replace("<=>", CYAN + "equals" + COLOR).replace(">","").replace("<","").replace("&/","and").replace(" * "," ").replace(" & "," ").replace(" /1","").replace("/2","by") + RESET
    return ""

if __name__ == "__main__":
    for line in sys.stdin:
        line = narseseToEnglish(line)
        if line != "":
            print(line)
            sys.stdout.flush()
