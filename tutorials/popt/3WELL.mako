<%!
import numpy as np
import datetime as dt
%>
-- *------------------------------------------*
-- *                                          *
-- * base grid model with input parameters    *
-- *                                          *
-- *------------------------------------------*
RUNSPEC

TITLE
 3 WELL MODEL

DIMENS
-- NDIVIX NDIVIY NDIVIZ
   100    100      1 /

-- Gradient option
-- AJGRADNT

-- Gradients readeable
-- UNCODHMD

--BLACKOIL
OIL
WATER

METRIC

TABDIMS
-- NTSFUN  NTPVT  NSSFUN  NPPVT  NTFIP  NRPVT  NTENDP
     1       1      35      30     5     30      1 /

EQLDIMS
-- NTEQUL  NDRXVD  NDPRVD
   1       5       100 /

WELLDIMS
-- NWMAXZ NCWMAX NGMAXZ MWGMAX 
    10     1     2      20 /

VFPPDIMS
-- MXMFLO MXMTHP MXMWFR MXMGFR MXMALQ NMMVFT
   10     10     10     10     1      1 /

VFPIDIMS
-- MXSFLO MXSTHP NMSVFT
   10     10     1 /

AQUDIMS
-- MXNAQN  MXNAQC NIFTBL NRIFTB NANAQU NCAMAX
   0       0      1      36     2       200/

START
 09 FEB 1994 /

NSTACK
 25 /

NOECHO

GRID
INIT

INCLUDE
'../TRUEPERMX.INC'
/


COPY
 'PERMX'  'PERMY'  /
 'PERMX'  'PERMZ' /
/

DX
 10000*10 /
DY
 10000*10 /
DZ
 10000*10 /

TOPS
 10000*2355 /

PORO
 10000*0.18 /


PROPS    ===============================================================

-- Two-phase (water-oil) rel perm curves
-- Sw Krw  Kro  Pcow
SWOF
    0.1500       0.0    1.0000         0.0
    0.2000    0.0059    0.8521         0.0
    0.2500    0.0237    0.7160         0.0
    0.3000    0.0533    0.5917         0.0
    0.3500    0.0947    0.4793         0.0
    0.4000    0.1479    0.3787         0.0
    0.4500    0.2130    0.2899         0.0
    0.5000    0.2899    0.2130         0.0
    0.5500    0.3787    0.1479         0.0
    0.6000    0.4793    0.0947         0.0
    0.6500    0.5917    0.0533         0.0
    0.7000    0.7160    0.0237         0.0
    0.7500    0.8521    0.0059         0.0
    0.8000    1.0000       0.0         0.0
/

--PVCDO
-- REF.PRES.   FVF   COMPRESSIBILITY  REF.VISC.  VISCOSIBILITY
--   234         1.065    6.65e-5         5.0     1.9e-3   /

-- In a e300 run we must use PVDO
PVDO
 220    1.065    5.0 
 240    1.06499    5.0 /

DENSITY               
912.0   1000.0   0.8266         
/
               
PVTW               
234.46   1.0042   5.43E-05   0.5   1.11E-04   /


-- ROCK COMPRESSIBILITY
--
--    REF. PRES   COMPRESSIBILITY
ROCK
         235           0.00045   /



REGIONS  ===============================================================

ENDBOX

SOLUTION ===============================================================


--    DATUM  DATUM   OWC    OWC    GOC    GOC    RSVD   RVVD   SOLN
--    DEPTH  PRESS  DEPTH   PCOW  DEPTH   PCOG  TABLE  TABLE   METH
EQUIL
     2355.00 200.46 3000 0.00  2355.0 0.000     0     0 /

 
RPTSOL
'PRES' 'SWAT' /

RPTRST
 BASIC=2 /



SUMMARY ================================================================

RUNSUM

EXCEL

--RPTONLY
FOPT
FGPT
FWPT
FWIT

WWIR
 'INJ-1'
/

WOPR
 'PRO-1'
/

WWPR
 'PRO-1'
/

SCHEDULE =============================================================


RPTSCHED
 'NEWTON=2' /

RPTRST
 BASIC=2 /

-- AJGWELLS
-- 'INJ-1' 'WWIR' /
-- 'PRO-1' 'WLPR' /
--/

-- AJGPARAM
-- 'PERMX' 'PORO' /

------------------- WELL SPECIFICATION DATA --------------------------
WELSPECS
'INJ-1'  'G'    1   1  2357   WATER     1*   'STD'   3*  /
'INJ-2'  'G'   100   1  2357   WATER     1*   'STD'   3*  /
'PRO-1'  'G'   100  100  2357   OIL       1*   'STD'   3*  /
/
COMPDAT
--                                        RADIUS    SKIN
'INJ-1'     1    1   1   1   'OPEN'   2*  0.15  1*  5.0 /
'INJ-2'    100    1   1   1   'OPEN'   2*  0.15  1*  5.0 /
'PRO-1'    100   100   1   1   'OPEN'   2*  0.15  1*  5.0 /
/

WCONINJE
--'INJ-1' WATER 'OPEN' BHP 2* 300 /
--'INJ-1' WATER 'OPEN' BHP 2* 250 /
'INJ-1'  WATER 'OPEN' BHP 2* ${injbhp[0]} /
'INJ-2'  WATER 'OPEN' BHP 2* ${injbhp[1]} /
/

WCONPROD
 --'PRO-1' 'OPEN' BHP 5* 100 /
 'PRO-1' 'OPEN' BHP 5* ${prodbhp[0]} /
/


--------------------- PRODUCTION SCHEDULE ----------------------------



DATES
  1 JAN 1995 /
  /

DATES                                  -- Generated : Petrel
  1 JAN 1996 /
  /

DATES                                  -- Generated : Petrel
  1 JAN 1997 /
  /

DATES                                  -- Generated : Petrel
  1 JAN 1998 /
  /

DATES                                  -- Generated : Petrel
  1 JAN 1999 /
  /



