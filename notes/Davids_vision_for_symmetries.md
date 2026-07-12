Here is my design vision for symmetry-enabled SAILIR. 

We have the P_S operator defined in the notes. It is a linear operator that enables any integral to be written as symmetrized form + lower (r,s) weight terms:

I[a] = P_S.I[a] + lower (r,s) weight

Using this, at inference time, we can write any target as a sum of P_S.I[a] terms plus one remaining corner integral to account for the 
initial unsymmetrized form. Then henceforth the inference engine (SAILIR's trained model) can work directly with P_S.I[a] integrals.

The original action space pre-symmetry-enabling is in terms of raw integrals; schematically:

I[a] -> I[a'] + I[a''] + ...

But since P_S is a linear operator, the IBP action space can also be written identically in terms of symmetrized integrals

P_S.I[a] -> P_S.I[a'] + P_S.I[a''] + ...   

So the IBP action space can still act on P_S.I[a] integrals with no change and inference can proceed as before.

Inference wins if there are significant numbers of integrals for which P_S.I[a] = lower (r,s) weight. Apparently this kind of cancellation 
does happen a lot in our pentagonbox setting.    

Ideally, we could use the symmetrized P_S.I[a] integrals at every individual step of each one-step worker reduction. Then the actions could 
potentially be simpler, if the weight of P_S.I[a] drops below the target weight threshold of the one-step worker.

  
