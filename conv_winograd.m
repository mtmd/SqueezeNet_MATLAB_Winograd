function [top] = conv_winograd(bottom, weight, bias, K, S, pad)
G = [1.0, 0.0, 0.0; 0.5, 0.5, 0.5; 0.5, -0.5, 0.5; 0.0, 0.0, 1.0];
B = [1, 0, 0, 0; 0, 1, -1, 1; -1, 1, 1, 0; 0, 0, 0, -1];
A = [1, 0; 1, 1; 1, -1; 0, -1];

[Win,Hin,N]=size(bottom);
[~,~,~,M]=size(weight);
even_pad = mod(Win, 2);
bottomPadded=zeros(Win + 2 * pad + even_pad, Hin + 2 * pad + even_pad, N);
bottomPadded(pad+1:end-pad - 1,pad+1:end-pad - 1,:) = bottom;
Wout=(Win+2*pad-K) / S + 1;
Hout=(Hin+2*pad-K) / S + 1;
top=zeros(Wout,Hout,M);
m = 2;
r = 3;
alpha = m + r - 1;
for w = 1:2:(Win + 2 * pad - alpha + 1 + even_pad)%(Wout - alpha) + 3
    for h = 1:2:(Hin + 2 * pad - alpha + 1 + even_pad)%(Hout - alpha) + 3
        for m = 1:M
            wStart = (w - 1) * S + 1;
            wEnd = wStart + alpha - 1;
            hStart= (h - 1) * S + 1;
            hEnd = hStart + alpha - 1;
            par = zeros(alpha, alpha);
            for n = 1:N
                d = bottomPadded(wStart:wEnd, hStart:hEnd, n);
                V = B' * d * B;
                g = weight(:, :, n, m);
                U = G * g * G';
                par = par + (U .* V);
            end
            par = A' * par * A;
            top(w:w + 1, h:h + 1, m) = par;
        end
    end
end
for m=1:M
    top(:,:,m)=top(:,:,m)+bias(m, 1);
end
top = top(1:Wout, 1:Hout, :);
end

