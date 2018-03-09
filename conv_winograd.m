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
P = ceil(Wout / 2) * ceil(Hout / 2);
alpha = m + r - 1;
UU = zeros(alpha, alpha, M, N);
for m = 1:M
    for n = 1:N
        g = weight(:, :, n, m);
        U = G * g * G';
        UU(:, :, m, n) = U;
    end
end
VV = zeros(alpha, alpha, N, P);
p = 0;
for w = 1:2:(Win + 2 * pad - alpha + 1 + even_pad)%(Wout - alpha) + 3
    for h = 1:2:(Hin + 2 * pad - alpha + 1 + even_pad)%(Hout - alpha) + 3
        p = p + 1;
        wStart = (w - 1) * S + 1;
        wEnd = wStart + alpha - 1;
        hStart= (h - 1) * S + 1;
        hEnd = hStart + alpha - 1;
        for n = 1:N
            d = bottomPadded(wStart:wEnd, hStart:hEnd, n);
            V = B' * d * B;
            VV(:, :, n, p) = V;
        end
    end
end

p = 0;
tic
for w = 1:2:(Win + 2 * pad - alpha + 1 + even_pad)%(Wout - alpha) + 3
    for h = 1:2:(Hin + 2 * pad - alpha + 1 + even_pad)%(Hout - alpha) + 3
        p = p + 1;
        for m = 1:M
            par = sum(squeeze(UU(:, :, m, :)) .* squeeze(VV(:, :, :, p)), 3);
            par = A' * par * A;
            top(w:w + 1, h:h + 1, m) = par;
        end
    end
end
for m=1:M
    top(:,:,m)=top(:,:,m)+bias(m, 1);
end
toc
top = top(1:Wout, 1:Hout, :);
end

