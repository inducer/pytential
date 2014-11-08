from __future__ import division

__copyright__ = "Copyright (C) 2014 Shidong Jiang, Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


def muller_deflate(f, n, maxiter=100, eps=1e-10):
    """
    :arg n: number of zeros sought
    :return: (roots, niter, err)
    """
    # initialize variables
    roots = []
    niter = []
    err = []
    nmax=100;           % max iterations
    eps=1e-10;          % tolerance

    # finds n roots
    # checks for NaN which signifies the end of the root finding process.
    # Truncates the zero arrays created above if neccessary.
    for i = 1:n
        miter=0;
        [r(i),niter(i),err(i)]=muller0(f,r,i,nmax,eps);

        while (isnan(r(i)) || (niter(i)==nmax))  && miter<50,
            [r(i),niter(i),err(i)]=muller0(f,r,i,nmax,eps);
            miter=miter+1;
        end

    end

    end

% Muller's method
function [z,niter,err] = muller0(ft,r,ind,nmax,eps)

% initialize variables
niter = 0;                      % counts iteration steps
err=100*eps;                    % percent error
x = rand(1,3) + 1i*rand(1,3);   % 3 initial guesses
x = rand(1,3)*10;   % 3 initial guesses
z1 = x(1); z2 = x(2); z3 = x(3);
w1 = fi(z1,ft,r,ind);
w2 = fi(z2,ft,r,ind);
w3 = fi(z3,ft,r,ind);

% iterate until max iterations or tolerance is met
%while niter < nmax && (err>eps || abs(w3)>1e-30),
while niter < nmax && err>eps,
    niter = niter + 1  ;        % update iteration step

    h1=z2-z1;
    h2=z3-z2;
    lambda=h2/h1;
    g=w1*lambda*lambda-w2*(1+lambda)*(1+lambda)+w3*(1+2*lambda);
    det=g*g-4*w3*(1+lambda)*lambda*(w1*lambda-w2*(1+lambda)+w3);

    h1=g+sqrt(det);
    h2=g-sqrt(det);
    %
    if (abs(h1)>abs(h2))
        lambda=-2*w3*(1+lambda)/h1;
    else
        lambda=-2*w3*(1+lambda)/h2;
    end

    z1=z2;
    w1 = w2;
    z2=z3;
    w2 = w3;
    z3=z2+lambda*(z2-z1)
    w3 = fi(z3,ft,r,ind);

    err=abs((z3-z2)/z3);
end
z=z3;

end


function y=fi(z,f,r,ind)
y=feval(f,z);
for i=1:(ind-1)
    y=y/(z-r(i));
end
end

