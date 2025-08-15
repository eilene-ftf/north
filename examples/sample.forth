VARIABLE DENSITY
VARIABLE THETA
VARIABLE ID

: " [CHAR] " WORD DUP C@ 1+ ALLOT ;

: MATERIAL
   CREATE  , , , 
   DOES>  DUP @ THETA !
   CELL+ DUP @ DENSITY !  CELL+ @ ID ! ;

: .SUBSTANCE ID @ COUNT TYPE ;
: FOOT  10 * ;
: INCH  100 12 */  5 +  10 /  + ;
: /TAN 1000 THETA @ */ ;

: PILE 
   DUP DUP 10 */ 1000 */  355 339 */  /TAN /TAN
   DENSITY @ 200 */  ." = " . ." tons of "  .SUBSTANCE ;

   " cement "           -131        700  MATERIAL CEMENT
   " loose gravel "      93        649  MATERIAL LOOSE-GRAVEL
   " packed gravel "    100        700  MATERIAL PACKED-GRAVEL
   " dry sand "          90        754  MATERIAL DRY-SAND
   " wet sand "         118        900  MATERIAL WET-SAND
   " clay "             120        7