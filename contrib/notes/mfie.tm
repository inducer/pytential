<TeXmacs|1.99.5>

<style|generic>

<\body>
  <section|Augmented MFIE derivation>

  <subsection|MFIE derivation>

  We'll do the derivation for the exterior MFIE from the boundary condition

  <\equation*>
    n\<times\><around*|(|H<rsup|+><rsub|tot>-H<rsup|-><rsub|tot>|)>=J<rsup|s>.
  </equation*>

  Next, assume <math|H<rsup|-><rsub|tot>=0> because of PEC, thus

  <\equation*>
    n\<times\>H<rsup|+><rsub|tot>=J<rsup|s>.
  </equation*>

  Then split into incoming and scattered, i.e.

  <\equation*>
    n\<times\><around*|(|H<rsup|+><rsub|inc>+H<rsup|+><rsub|scat>|)>=J<rsup|s>.
  </equation*>

  Use the representation <math|H<rsub|scat>=\<nabla\>\<times\>A=\<nabla\>\<times\>S<rsub|k>J<rsup|s>>,
  leading to

  <\equation*>
    n\<times\>H<rsup|+><rsub|inc>+<around*|(|n\<times\><around*|(|\<nabla\>\<times\>S<rsub|k>J<rsup|s>|)>|)><rsup|+>=J<rsup|s>.
  </equation*>

  Use <math|lim<rsub|+><around*|[|n\<times\><around*|(|\<nabla\>\<times\>S<rsub|k>v|)>|]>=<around*|[|n\<times\><around*|(|\<nabla\>\<times\>S<rsub|k>v|)>|]><rsub|s>+loc<frac|v|2>>:
  (where <math|loc=\<pm\>1> depending on surface side)

  <\equation*>
    n\<times\>H<rsup|+><rsub|inc>+<around*|(|n\<times\><around*|(|\<nabla\>\<times\>S<rsub|k>J<rsup|s>|)>|)><rsup|s>+<frac|J<rsub|s>|2>=J<rsup|s>.
  </equation*>

  Rearrange:

  <\equation*>
    n\<times\>H<rsup|+><rsub|inc>=<frac|J<rsup|s>|2>-<around*|(|n\<times\><around*|(|\<nabla\>\<times\>S<rsub|k>J<rsup|s>|)>|)><rsup|s>.
  </equation*>

  The interior MFIE is derived similarly as:

  <\equation*>
    n\<times\>H<rsup|+><rsub|inc>=-<frac|J<rsup|s>|2>-<around*|(|n\<times\><around*|(|\<nabla\>\<times\>S<rsub|k>J<rsup|s>|)>|)><rsup|s>.
  </equation*>

  <subsection|<math|\<rho\>> postprocessor derivation>

  We'll start with the boundary condition

  <\equation*>
    n\<cdot\><around*|(|E<rsup|+><rsub|tot>-E<rsup|-><rsub|tot>|)>=\<rho\><rsup|s>.
  </equation*>

  Again, <math|E<rsup|-><rsub|tot>=0> because of PEC, i.e.

  <\equation*>
    n\<cdot\><around*|(|E<rsup|+><rsub|inc>+E<rsup|+><rsub|scat>|)>=\<rho\><rsup|s>.
  </equation*>

  Now use the representation <math|E<rsub|scat>=-i*k*A-\<nabla\>\<varphi\>=i*k*S<rsub|k>J<rsup|s>-\<nabla\>S<rsub|k>\<rho\><rsup|s>>
  and obtain

  <\equation*>
    n\<cdot\>E<rsup|+><rsub|inc>+<around*|(|n\<cdot\><around*|(|-i*k*S<rsub|k>J<rsup|s>-\<nabla\>S<rsub|k>\<rho\><rsup|s>|)>|)><rsup|+>=\<rho\><rsup|s>.
  </equation*>

  Carrying out the limit, we obtain:

  <\equation*>
    n\<cdot\>E<rsup|+><rsub|inc>-n\<cdot\><around*|(|i*k*S<rsub|k>J<rsup|s>|)>-S<rsub|k><rprime|'>\<rho\><rsup|s>+<frac|1|2>\<rho\><rsup|s>=\<rho\><rsup|s>.
  </equation*>

  Rearrange:

  <\equation*>
    n\<cdot\>E<rsup|+><rsub|inc>-n\<cdot\><around*|(|i*k*S<rsub|k>J<rsup|s>|)>=<frac|1|2>\<rho\><rsup|s>+S<rsub|k><rprime|'>\<rho\><rsup|s>.
  </equation*>
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|1.1|?>>
    <associate|auto-3|<tuple|1.2|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Augmented
      MFIE derivation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1<space|2spc>MFIE derivation
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|1tab>|1.2<space|2spc><with|mode|<quote|math>|\<rho\>>
      postprocessor derivation <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>
    </associate>
  </collection>
</auxiliary>