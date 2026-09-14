"""Kaggriculture agent — E018: ルート農場 + 全品目リアクティブ市場 + ギャップフィラーハンド.

ルート: 上位チーム (ИТМОНИ) の 720 ステップ行動計画を base85+zlib で埋め込み、
farmer/hands はルートを再生、市場オーダーは全品目対応のリアクティブロジックで置換。
M3c/d: 余剰ハンド (最大2人) は水やり/雑草除去を最優先。ルートが今後48ステップ以内に
BUILD_PASTURE/PLANT する位置の雑草を優先除去 (位置シミュレーションで予測)。
種の購入は hour2 以降 (hour1 の雇用資金を守る)。

検証 (2026-08-29, E018-M1/M2/M3d, ローカル・決定的相手 base 15試合):
  - 平均 $61.8k (seeds 0-14)、勝率15/15
  - 旧リアクティブ型 (E017b, $41k) を大きく上回る

提出形式: このファイルの `agent(obs)` をそのまま提出する。
"""
import base64
import json
import math
import zlib

_ROUTE_B85 = (
    "c-rM%U2hyoa{Ma`KJUYFhT=+o<4Jvza9mNKC=bpHVX?rSVZb;)to>&AzgrT?nV#;*jEJnNp?2?$8j92PkzJJ;85#NW|DOH(ufP4{"
    "@4ub>%XdGYy}$bK;q24*-<|!(umAP8|NipPmw)`@*Wdp0@Bj1lU;p?0cSnElr;l&1U;Xm=&+b0nPT606`LBQd<?7wrpRTUYKL6&!"
    "n`wIg`G@n*|Ly*FAExQmm*4y}UEjR>{KMtyH!rVlZ*T74uXb;yt1rLVe*MjxxA%i_e_BoN{o9v+`S^Z0-oqS!KHE(n?(V<B`>Azx"
    "Hu35_1M|4PdO01wcrgb&%lVk!-Q3>2IeU2IV#nFsR)hIG(CI8(#@VCPAJJOAzrA_&@#WpzHZJx{c39|_H@mOXO?H0%=SQoyTGOA;"
    "rmM$ke7O1eILgC^pa10DY)J7OR>S%8mrY&_;Bg&~mgF$*$Eih|c6B%1R+jO+wv312IQx!XAa}T{K|Oxed?5EL^z!EBy=Z%AFZZ9_"
    "zonH;Ro}UTQO!2;;Yae=$3Ml7UC!bA`SI~(6X|Z%H)j7;^P$bZe>T-gc&YfFIHOyxLNrKp=A(gYyM>G-*rWKfhr40iD+SN=m0>P^"
    "gu_$jMwX13(d^xs;<u=6i{RtPe$4KShaU?!scr|zChZ}tJ;rD)XJ7sJN9l?b9z6Tr{0WC&uLs!axpO<;f4EwMUX%1DKJcllX^))Y"
    ";-_!uQ7iB9nG;p54;1?#Ab+rSfu4LwYq!Kd=AV2L-{P2F-dtZ#FYkW&^K^Un_WJF=-M^pmpYPACKkNhTm!a`_zLD%t)p2w3AF)2<"
    "&Qk#B!*u;MSe&iem#Er)BSOD)4GVp^yS>`|X}Z1r>ppO8M1*qi0{ygJrfrvCi5a1eNNErRikkE9ecZ?fT!_R~2iLkif}zXVj7&*O"
    "O27KBs@g+fG_*#47C#=qB^U#>!Yw2;CYVB}InL};iu1q`8Nm~9CA9zpTz^p<HTx8sulh_lGYDqRpKc-<2zP=o);vmj3}R*{`RvWF"
    "ObbsY8%PCDgCX$YA*;tVtsRO}<`9??vIy4+ifHQ(gm&IHxO{bc^WJ;H_yC<eA#jtnpHA^;0}Ji_aKXF0^u5b2E}sn3A3lsu<G%SZ"
    "`Mzwut)uDW(=53}Jr`R#S&7%lfc?hR*6)ri4bq5*56e^FnY_8W{a3W3`_BtIcJ!Um13IKUgTPXBNXekck|!hBXs6cE!bCe_`#)$u"
    "pzjDx4tjVMgAQHA@E0vP9~N;yuWywUY798wH?N*d9x~hf;4yd$m(J{vux4dPm8TF^#^I-Xf3fWmk3J`ULhkbIDLla4gO7zdH8@@Q"
    "!C(I__NDZz4LRR%yf`3>Jyi7sL_R7U05<jdR>=<zPp0}hNK_`xlipA9d=p#6_R0O~DwB>a8%(ZKrrcnfk{V)Yi)C&}gIlk{=5yRk"
    "-0Qld4QN$M8W?jKW!Gdy72IKWwF~1U_ITD0t!y0QcZi_llwn#Br@jBF;x&a!0+HJLpK6X14c|0F3I<$BNL1E~&f?YE*Uf=vLrhy6"
    "bc1h?`D3Q9%|?w}#ySX%T*596^$EThJB0(kTJrsKK-qC*Wt6w*ygmN!=ZZrQU83#p06X<OItXoEt*aP%6dRm>axd+gL)YelF`)CL"
    "?id5bx+H}K*N80h2uPH`VU2n4+|O=6MO_p3TvRbBpSxfZ_}2~2m#Gect=DIa7~Jv{ZLluD?aZwsNE*bkT#pM`WI$(#-LS&RI`=HT"
    "c*>EpgdWf_fs|Gb^&lbSuAsS)WhYkm4Ra6f0Nx&c>QWLR=f>e90tar>IWI4{!2S~sOKG^mz|>*9EiP0CSJvl0;wk{G0{HdK&4<ZY"
    "hxyYEf2(lc3NRshPiNqX;6L?4Nzn_9HX|C9OQ+0(i2YfQII4;Zy^r5+Zmti|doK@!!<)H%es-o_oq_t_#9HB*{)mU(|D8f0HWT8R"
    "LV!uGgJ@5caI;B@r_*SB<3j_%B*5w>FQK1gnx8%a)w2iRgjJTWOx+e`aD!Ii+*`Yw90gvyDdHRx2fg4ivv<H{QWKAvQPRuKWODws"
    "VD&KU&i!E<Oh}YgwCqM5wayCPX_2=X0?4)04NIApO=F^Y+X6DOP;M&nNx||(n2aj`Bx2lS6}<+o`m;}VDcdRQ!8*hY_}WIFZxYsA"
    "@Siz|i~$E@{1#{r|1<IgIvf)#f5uKR(1dCk4#HVxXVGj9c78^+VoY$ciDQ=#A$utyby-pj&ep}7bZ;k%Fg(<}O07kEjX_eL$0z<k"
    "J&Cd`ER|N#%tO?-c2zp5WVB3@a<8q0L766sL;k2R7__ZgCNNG!R5?!YtfDUhIwN${pdm}jDjBmEP-XKVXW+19@M{rxkCl<uoY;pY"
    "thm<8NHKaAP-cqV|5X`cTP{`|8QeKjj)rl^nMxBnB`~46Jen-u5+^Jk(bi>(p;LwEDmS7N*ZEjvR!+GLlMD*GAdSO!@#<bPfO3rT"
    "RD$ttd(?f2VQ;!@kx6WdSu%=F2N?A5fEjak<cRq)^YoyiYy7#w3Zb}yn!H1<Prd4vFn${9wGM+r;<I_GONq?uLxTZnInb9Yn+x~L"
    "Vd<P*7qmXIV5o^xL^fOI3re`e<N5g0+v~sFXF#=60wWT3Ewx;To8A{FHR|T~@+><;yBPA{RsT83|5nZs$}~%{hXz#(n#3gW4m{Hm"
    "X=ytXLkGB08!}N6Nn*J#l*o8~0_a6FpzX9PN$4@4!Z>u~VxS}qJiB%g^rYp?gm}2Kus4GQcu1ZyE8Ww;I@R(GiKtlQ^nh<{vU{0F"
    "T7C!N0TPgka~o^1A#fc=Wqm6q5rOZO1g~jNlF1OKh1f>kL?V;hXkfFW6<-!fhK9?i)zM4R`RG9a1B5#$9v0kACM<}?bJ%-4oM|mC"
    "0e{`BQeBYPdfqW%zoGu~H=M2CI@$WI8J^;NeR)=px)r^UJ1VRc%_Gk))$_eG^78%+66nuX=J&KLW|But<+27&XcT)Old~-rhBmH3"
    "2i6H^ArMvsVM_Gip`gr|iGb0Za08LD2NT<(9d)VcgsvOYded{Wt~5f4lR$YLlPc&|&;R2j_SCkdS@~rfM}a)Uh`{USm}7+GH6Ve8"
    "XsrY2<YZuQ5tH0`;e8>|7+Wuy456upq$NjQ1!nYe)t4_Xz>+Bql={|HqgfZuqp|Iv5hB8YjvAd{fLbBzanwp8*{2%dEQVJW%PDGo"
    "sN=hcWmqmg4V#TL@(xTD5C}YPm$FHbT77*lN5KWKGh-tqVJozZDZv8e<~AH3IM%N-!Zl4<5ZzcKQ+yVgJa_$d)%FuMQ59mmaeVp+"
    "K^mewlh?Ql4zi>wZZ54~ZLeZOyuw}Jkl@hD#hSeB+%y?+QX$IXKc#$Suw@!SE2YAd-}k6!DD=Q;FB_?wk=$}f&C@HJgJ-A;uCf9g"
    "wVF!ky`Wi9iz)zN>MfZu*68(?+lu$lq6*E5FOG%{|8tm}4)!af%$%UkkDRUsBqCHj1uu@rr6Z|q=S9vDm0-hGyFRS*2owa%N5nul"
    "%%l>mSE@nn)KWz`@4Oas1det=JShD;@$Z#t(9C_Axcpp;sH%>K&jqCcN<<qEPM+wnui-_I+;E98zQs3$=)XO{9nX&7j-RIMn|FJt"
    "L6uX>%lYI~rQY#gSM6I*Eo(t>I3w@nsWdMnj}-`hiYoEgOp&^taqb^)GNSrjeyXc?H_aTiyixH{Z-bYMu?-jFq7VMKRu4lj-udyH"
    "_fWZzypG4J+|%pV`_Fz?m?YcO%_*5$4iO_lqzI3kxxiD#ZS0F6Si~om&2*q0v_wQ=td$7m>7j5I&Sd7`!^(b7`Omdtt9A|9V*?#3"
    "^)j0jT8emOy#9i=2R79?`Y_C3vF#$?BM{bmSQ|+U<YcyK(hBuTqX;TlNm`?AGcA$9W)kSlvhP8Pc2Ty_Bv8U?Ai(_;g~hWUXL-Lz"
    "^SZy*^%Gwyf=u2l?7Wiehrs2;<AsP&GXHK3AeczA!9q%vpdO~ssaqQ!OQ|jdEpnHB0*dadR+!Q(95$PpmWyE%GU%j|<gV`isA81+"
    "Dbof(5L-wueRg*eXI?CfCoOml<|J#QOHYQn)~Fl+%$^n}X9&wVmSRz@X8(s2t!Jx;?#2oMRvLeSy`Ez+G{F$#GJT`XJm&!+Ky3Ik"
    "SSS_jaru9hw44sz5sXtccpaN1lvRS!L1RX=KMBP>MG@f(fQKDZ`i{Y<CuZ{rh?-07i$2ffdI$0Rc(ob3dAn-rw06IM0bh`&jlxe?"
    "iiwh3(xi5MuB@&uBv9l&V>C^R%~u9-r8rz*n=(_aYG?NXa`}2!O5vPdI77-<p-n=UKo3{w3DwMnA<QKh<9cRQAPnb%Sr55kspD%8"
    "dxd-Gkp33nX8zBP70Efg0i7%F;AC+*WdotiV3!@m#>f;&EOk0U2`eNj+Paac)sPIKlCGhua-2$m&bn<nEk^kwxK+mlcQB~5vpHIH"
    ")h_*k*}=|W&<5gmcXAwc5@gWOz^m%Wc*VQPQ4_JcxT!vv(b}k%P{bWAw@_lW6Qy(sk|?qywF07-MK!EovCLjkA+^++sc<V99BEM}"
    "o+4MB`b5CB2`~XF{3VaRA?>4Ab<<8PZh3Q`Hi{ma#wlknLdw|^Vmtn{Spj?q(#Ey56u#O37*XyeBGlpu=n^_H0;j1X%*JjQ{4!5U"
    "_a_NGBgkBngb#CB-OY+F^(dlr+jZvkdI()w1lARC8kr3v3uo8cr!Xgvi_%-t>a!`%7P6Dcv8$8pQ1s@z!bh@AInidVjL!wiw_Pft"
    "qC9|8-Abu@74lH2Nf*PQ^!lxCJ@1Gd)nu4CiNBPl>upJwk{WmrjEE_S3fz(B#?T}f0>WH=T|%($h9IJ?2Vq(_KRthw(6_<gl94ES"
    "Xz8+2Jv0(YA)h0RND$i&{u60)%Eb)OyiNcfqt}A82{5)mq*ONBJ(>lsblr#O#06~X*HEDfR)$NUxE>wJ^HP7E+NBmS$2|_4k~>5s"
    "PbtK&jk90R6C}b`m;!6&XSSE|&?hd!c77g%N|ykcI{)F@V|O8jc%9tU3ybPO?rP{Q&aj*mj2cDs+UYu7@MZECQ>}ABw#nljPpVr&"
    "r(Bv>*}p>2M_b3h7SL))3_=J93vj@TNfd#);;Xi?&C{S65cJNfNbaPghb*ztjx^W;Y*uHo8UgGa34JbQRZT8z?1q7aup<@B8oKs9"
    "nv@7jX=wXl$((uNriw~er2h1fJwENM?v0BK=N34Nml+98wJ@3nwz?(QW><+V^X6fOVxbJ2YAp8h1?gdjI>*ChT{SeMgAtjIaqG34"
    "9RhxRtbAXnoLusVwsy~O&GZQwdTMo@H(6{dVNNNbzeKmHqZYDql)mQDmQ|L#(n*iB*AtmUZFD6hgGUto8K*a6AWG$V+l&$9ZU>nR"
    "Y0IF)U}8JQf+*NQ*W$yQX?nlvN$}vRyig&pqqY7qEQS2T(5|vIJP{>}Ip>FI91qW;Wa-g)|3oz*w@34%dc#Hos{yc|43)`xy$u>!"
    "CbdpbyeS|QQs~R&jR@hTt&#$HKlkbqpHKL^kEtFLF(?VLY~yLkw3J9kDcY5KKg%w43lr#OPbImk`u{x%DrRLcs{r<}*OrhM3Qa|i"
    "&tDK;0ZY0Nh;>Kq4TFFPE*%g-Vba@ez}{woD?D)biZb^Xr*^k9rqtm{s(s}(tPoNGif&L>Ne`%3k8wOb(adu8CXCatiksr`;J7ZY"
    "b0Hfw`(r)@5+d&iVH}mx!@X{L@nL+oxNvh%#iC@BP@I(>Rzg}7fw`o-;FijmY6=>mmQR;ZDy~$(iW#YzGOfXKrSLDPbWlYIzoym3"
    "%9&@M!qiPeE&z#(&Tc4*$?l9BIxNu6GdM>t+3;=Ciz@J57cE)pR^uiWn7C%-KLH<-$p(wScyV^<G7kjpP<;Y)SD?Aq+Y$}H3)gol"
    "?O?hG&whySspqWc>G9k3qOaV3S8cS50i~Jc_5qxe<S3(9ttazNVV$z9uwf!|EiJ^X+&*z@%bYp=X5B3k1mFxwxwb4P3apVCQUF&V"
    "G}bvUIp#;o4!G`1ZHw!+CpV*zmDfmupSoEmXznLc6RmSCDT}Q3Q!79>F>Dlrv<aWALvz0}tm81M1jzt==8x&(3mtPWX+}I(sR*}&"
    "kx?O?F1;~iU$xcFVDQ9Lm-K?etYUxEu|{s3Qe9X|{u`FCwXz#bRfDkSU%h?Z$;^l>8E=*fVG_Nasn;2&n)D!M6nR{RV5BU|=hS2&"
    "D(%MYUoNK-=$Xgl+Itk9FeW^YaxCZ?8OHwQ3bWGA4a@aB38T<gpUeDIWDUH<Z1TPZ*%O{p%vm2sI~`ZHqd>Hk$ttQU+-_<Ys9RnF"
    "hMKL?49gsBf7p$a^iY$|^DSfUed8>>3mJh+VEb<nlH;C|RpdZnlo3oegr2ZO6m#&rPJPW0pNyD5UIM>dJK%7`N<N8lHfdiJVG&xp"
    "&+&#xYo5ArDs`FOR;#3Sgav`$3{z@u(%GO!g`SOd(Q8svL@<4naWbjKA|K8}ZiZnnJk#sc9*RPt^>-`>VsweB`1q4?o2HaYE{P_h"
    "#xaN~&9u3Ai_}oFFxUYl<3;G$v4imbty}~FFDZD@Lc1dFQCUWh?vl{bqS@%2-Xff7uAdkj3@GeW06dV+-E#7;a7d4n4|EDe<X%M?"
    ">#9O<D=48EE1$-D6(B_&ZQ9CH?ru>YCxfHqrlrss6o?h~06!V|V-|qsu6g72bm$y9Abm+Cvf~kGw9!C#)DRTO;8B7uX}&-Tr1LpH"
    "j=o#Bs;VBHN|S>K->^E=r-Syg&+{x2SjnN)xE$!Sy0GhH9pFSrYqB*>T9!gFnjH$qtk`E^m`Yd=njTY3ak0UcQY%uGOG8HznOKcg"
    "N0?t7X4hNa_7Zw?7=5yta#~^v1sK&7eauksx}c?6B0owdau@>!)JTRnChy^?6w6x^+B8&%c~A!Q_6&8?F{~bX<YmiWW5wVvvD)J@"
    "EgQ9Cvmx4-)w-EnpKe&QuAyOT#Fn(x$G5aq46xh^odEW&&=brIC<hv8y$~u=4(>+a^b{7`k4lNVsftCh+CknUQPwl%X=^ttSn5}5"
    "ed=j2$bxw+?W|X}c>PUAsOA~B96Op4bi5+lLjWRK%2F3KZ0N2bw`}F91xB>7b^cUSV7yoM`{>lR2Rs^WTl9c<fZLwMth_%k@-_@_"
    "8;HCWCDEv-L9XNz18)$;A$nv)M}bc8gfJVqSc|?GF`2$`OU|=%Wyb}tf%HB+nHAy*l-tsQ+p$m%$Jm{aYhnmy#b_2l+&U@Zl}NLq"
    "U;qsb`iQDCtC89g6M2BH{IOhm5xyBTw%XZ(tEVlq+*xp>;PdHPjiV|q$?GJSVNyMLEozXJNf#Ty;{_JM<>geJL1m9~q4HkBLZe4I"
    "@lVtB%{wE8ZdaA`d{DSCkl0XZz|>SbE-OaN*%$jj0#Ha87K6^s;kBbV@m^7dFe3DZE_NMSXe?$BBeNJI;g^vjj8I1=be$|-C(3vS"
    "%PPaBb&orC+B0Ca*G-kIRQfER2*c25XwA94wiX>4F@gPH?vluP)YRwnVq$#Oq8S<c#g1dddu%&vdv>zm<+Cd2_bgPRTT0y-Enmf+"
    "Dq|Ju20(UYFw+TdU{Q{a)<rtXtjR;LT&4n2URi0?fv~K_>a}f$B{yyfpHtOL@gBi<27;lds$0xi6&5-zF}8?o=ztkIFDY6Lv=cOb"
    "n^Wx{6&>7?l1-2LXBICz^``rosCpyis%JZ<kgH>lTMA_UPXhyT<0$RtC$AESl=aSH_^T{O*T@tiDIE!b6Ed^5>ZV+?N?d<KobBk1"
    "V|n|wE&>Sf-9K6v?Fo8%NveSz!#lXXXcSz%qSBGxmLo?MIIGb*)nhRx610m@I!#Nww5kC@ZDgIc!zfC7Tedq)=cqd~eW#ivH`UY$"
    ">G<wIr1YRkrBW?jMaLc*U;53?ngLP%(h_;-xB<+h$mNiCygfK@OPHmw@=&#fK!7})3f3O=F>SSflJg*x4bqztSMZ`YJGCV@vt}~J"
    "!4V1fauUjs3$A}~9WgR-uB(Gjn4}S|?AaPlJ@6N;*uf_GZDf+)wg(y(V7e~$xBmQ@{rO1&!VVERsfcx}O=aD^Dl%x)MKBhEPL0Y$"
    "_oRtzOhAwX5~O)YJ$&e{4hRmTT*o2rP3jcmSQ3z(ntSGE$XReT6&r1bU<q8eHW->hVf;T2IYGmECF(+dQWL5B#vpXS+c<9#W=pFY"
    "E3MNY`kCuM3T!h;!A3k#E-HxLP9zAjG9$lwwR(^+<z!lNmMG|PdRjv<4&*E|NllB%gQxD6^vU5CZ8=RMDo5bIS>lww39xDu`4H@u"
    "G&Pe2Wfgszkq=07-LPvGfsqAuUYB~cV`BDs-z=C^5|GlYu3@CjS*jFfFJRLkV=P+XpkYBxN}GplD4}yMuWxPZ!)fiDSE<c5YiD@8"
    "+GP}+2`F#G;-nm4CWk~r0#Cj@pmJc^f@Q|Io3utAicM|(&b9Fi$iy#*Y(WxfNc_4B1WsKd;?-;dM*kmD^N=*f2>*Jy<%24FAA*Yo"
    "diRy|zw9x~J7-+7&KXNW;O1!ye%GIG!_U7JZeT&GtlsFT2~d+(>~egKFNsy_2$Rga8&xugK(sdM&Qub6)viUkI6Y5>DcVz2CR1Ow"
    "9P7+n@M(oVhZ!qFH}M3O>kL>E%81dGnKSk}6%Gsi8aaFxeIC<iIy90~tBDCr5RI{6Ioeo0Ia{9e5`m-))`rQ{eP||T5<x+jFj^cO"
    "B-B{^Z{OY;0C!JSi;w{`WUE&pF`{^}mXJ6BVl-!OE+FRK9=bU~Gdk?wf$4*jAX^C#9#*kxFe!+dC)@4-AwtUvv6q992o2gtFiFr!"
    "n&=M_07bYE(zID{0<gYlc@TSV6&F@6jU3LPp=Nye%6*Lu5tn9RSA3V`h*jpLMsRfi2=$`-GL#VYJB=YLNsv~UX<Z>vCFklbSs8dm"
    "aO)xz4O5_UcN|pSMCX}k>(!uUxw+J`0xcs4Yj~-W%_%Z~8Gn(P4ke|lHTI`aDtv%_pVM+>03Cl*I)x5w`3=IHT|2G12v?74Kf9kw"
    "uWp!n94MtuPK03+Zkg5KK-zAX|5!ti`gL*DCEZL7L{ENhter}m<zSaWq!E3-_ajbf=!BLnCxZh9VqA1MB2-lS5-a+5l<wxViD;?{"
    "hqn%IJS?`7@7vWok4p1~Q)XjT>?Lxv$cGWPqVS6)7LmnlIZr~Vo8C=5mEtI**tOJcxcViln^owlB-ZOF-J2uU#xCViI=`X2mEh-f"
    "w@M@KTJwUn5Q$TKp~}ILmKUZBD=FLWFCKv9xsgbY0vY(b*Hn*=cbK*2rQ@AMqL-V!aXd>R3?kx8eN`GDZh%^2?50AS<ij@sh|NT9"
    "(3ODX^A?tjh;g(eUPVKjfHTyXmvkioh2xyY3BNPD^M=fa3Lvs{1=`311iw#l4M71hAnlPDpSvW+QafQOpn3{m@b^*TY(`=o8j@P%"
    "<D<pu!(yygZSRblgCR|V1373G!3iWOcLNPO=uT#=_BP&mY|frMl016g4I6krWWCJtG~8N3SCsonq;v_Z-^oaRtrv((!gor%H-c9S"
    "JzUnw$tyx7Z(PwE!)0e^v;oJP)&OD`GNh<z#+H*Vk+E@UIQSMpIdQ4Jyhyvu&u)Ua+ucU&^y0pFiO}F0aC}6-CnPrE_{f7FezFoU"
    "xfnwS?Lu>@Hm?CobUUsryKufkhl-?$&BC*sz}8i?0`o6XqcJL!EdC=#1M`ZLQd@;W!D;>i!u{*7;~G(-Ncm<EVz4}Ceq#6;`eq4K"
    "r(;!1&*v7_ohp7x-hEU_jQDpdGm+{)_moqVTvdclkvl!;F$Kt^JD<Q#`N$T5o$NW89mM$9Sp{+LqHyM>&{IcmR<pz&3mTRyi1rjL"
    "#zk2*yjk}vh?=o11RHr}{I(9@b!pSmf>#bLz9w9RViP+-WUIAExl^wMs}HfI`^1pW*50E=n>tjNCb*IfyY)1*Lr1!>>t1%?9~J;~"
    "pE~Ll8Q18D!{AHCuqE=+3N=LGvNeM+IZZZ>-{_CgR_k0sxnL9p<eVDiH(_tAb<Mjnv09L*_|5H_*#tc1b>Fd3%h5G8xDjOAoa#xX"
    "gO#8~>H;tb&nJ;1X>vhYcW_!~o6Z`vM+6^6WEL>_e4FAF(-Qh=0~B+#4XY^>8zZf<997I?N|m*vK|z0kAl@;w0a%dt4`>u1wEXO("
    "7jHYaxM!-13%oF=g&ZGwqr4_2m0Y#!W>NV`rXp~?NModWl%>UJn2Iztn7A%{@5VIuim!dSY=^-69lGRD5;Xd~EM^k&HR}T=9j|kl"
    "-aJo*PC6I86B?@|ns%u+BE*9C=6Tt-9QjlrU3)*30gzpgyi!{v*BN%LmBvlF@~B1vB3l9*#;qW(qdL$ovAleBt6cT6BHYtm{yb-;"
    "u&t$%4O0g;sOXQA0dO6D0l2AGdl#cXh^NSwD;TTUmllMb9Z53`lS7L9LZ#W1mP0kUcPP}gL^)Z{H%!EFJ1Vq@p|czY6LWOhPSM#g"
    "<bjIQGbBsui^4Pp6JN7um^7(vk^*9QE)PLMFCEU#$F88LjKCb(7UXy~0NH^MAB5$M*1pF9IF_#41lvRCoZHOfmzu0O0ybWaDT8NY"
    "^oCB_1@Ht`B|RgA6sLK-t-`v?P>51xLPAfNpA_7WAe))h)pxBy%o%rlx41t<h=f9Qk$!ZtW?4$@3(<aL75Dv`rW+V~j%l%*mX`-<"
    "DM8wCNv5%LHISUe9&f4Y8NnCds6D53AGH^%CqPcHzEZ20Fw*G6QRqYD+hWe%3xpS54CCw#E<)Y`7wIWzd?p)8=+mVX7h4Ru4v-&g"
    "Wk~z5S)=3!c3;ql5Sw5{nalI))%ki<ZECK)qLh-Bw%Q3xt=57K5`wg##BY&eRb6-Ky^+&T!`K=@X;xz=g4j3ff2qQj5YFIjUL<;7"
    "UjYM_gW!CkYBNPsq+-PTr`#bz=V77Kji_@s1n#9G?}P=E#{h)F*%USh(61^yCxprdq)j%f0Qxe*N51fKCLs~^>RQB~m-U$vU12BM"
    "NaNP5E+(a80w%~6YG@HWuF3mK+h{2?HmJ5#{-3Ga8-iafsHJNlCMVVMRv3e`!f~mA28JT^NvY66mPqhm1lqNP%tH&UvFe)AR3twX"
    "ot`Vw=utd^ME&%^FqqJ7C)f0bU_w+bugIo_`Z`6qB;;gte>o2G(OJ%U&8%tx#M+vAb_EAEoGg@;L6d+Dn;90LPwfZ^+yKPyvLEWW"
    "|KI0`beNR`pZ@?x_WT4zv?F_(JQWr>K+4<{wj-@Pi#mr5!aBwJwuY5y(s6<80#mt=s!)29&^60WwP`eZ^jZvb+8ejzq5(mcb)<)3"
    ";?Mzcv2P(rO&JxfSz7L=bJ!LErX$nLd7(#;U}YQEN4}jDCgLAWfy=Z6Fc&^5EH)=)MNk(zj9F{wT{(3x<AOV()oJLYf<-7;hjHY$"
    "40g4R1yjIZ2Ll&08lrQRetX-(9-fSO4pT~lpmF^wpIz^!d_hTUS-y*6y?FdgfDHT^u4vG2n=A@oa;|E#v|KG56s|<w|4;JHDHbhF"
    "?8lfsvW!U43cPTen^?|C>6UEk(RdL|vmn#95@~hoObL?2>bvvRahq0=v9cDvL04&nr+~q(-U~8V#EFKFg~(9k6GIG;COt`afJYO<"
    "C#=-3gUk#bDiG8XJbh9&tCO11L|}O?`Y9+j2Xc<^Z<b$Kor6&m+|(bWFUAzOy@V&yj=JbbT1y{pUdXs>ZIY489{aJbK#Pz`Ji(Sa"
    "qH^WxVon&FI$(QgQEO{I_j->)y_%~K0F%pCqGI-{a28-1DUzdNu0j0HUs{0*8hkf>Pz8oGTEuzymMkzW?FaIigl*>#1mideHm60~"
    "j-*focA^}ounLZnKtU2(%~I!HG(-{HhD)_~F)aR@u5aFjMRINl3ByKI{T(T1R=a9Y2rsa<CQ%y)g-q_qRcIS}-t@{NdG*;J<tWD9"
    "p+;cEk6A+$X~G11N(LOlQ0&i*SOmjS&us1oG8ZXgQl_^2$^ss>h?<rFjL-joeM5u(oM|YH)=Ld56ZS_S-7a9UCQB<ibI@!M$EW&J"
    "FKXG-4iox$>PX<r^3Og{<bB3RNq*4u8zx{;<(2!N{vWnavpW"
)
CROPS4 = ["WHEAT", "CARROT", "STRAWBERRY", "MELON"]
SEED_COST = {"WHEAT": 10, "CARROT": 20, "STRAWBERRY": 100, "MELON": 80}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
SELL_ORDER = ["MELON", "WHEAT", "STRAWBERRY", "CARROT", "MILK", "WOOL", "EGG", "FERTILIZER"]
SELL_CHUNK = {"MELON": 15, "WHEAT": 20, "STRAWBERRY": 10, "CARROT": 10,
              "MILK": 10, "WOOL": 10, "EGG": 10, "FERTILIZER": 10}
FEED_RESERVE_PER_ANIMAL = 2
FEED_RESERVE_MIN = 6
WHEAT_STOCK_TARGET = 8
LOOKAHEAD = 96
SEED_BUFFER = 1
ANIMAL_BUFFER = 0
LAND_BUFFER = 100
FERT_BUFFER = 250
SEED_MAX_BUY = 8
RANK_SELL_ORDERS = False
# M3: ギャップフィラーハンド。ルートの雇用数に加えて最大 GAP_HIRE_MAX 人を
# 追加雇用し、水やり・雑草除去をリアクティブに実行させる (ルートのハンド位置
# ドリフトで水やりが漏れた作物の枯死を防ぐ)
GAP_HIRE_MAX = 2
# M5 資金フロー修正: 種の大量前買い (96ステップ先読み) が d0-9 の資金を枯渇させ、
# ルートの重要購入 (d4 牛・d6 土地) を遅らせて動物滞留・土地遅延を連鎖させていた
# (LB 実戦リプレイ分析: d6 にイチゴ種23個=$2.3k を前買い vs 元プレイヤーは4個)。
M5_JIT_SEEDS = True
SEED_LOOKAHEAD_JIT = 12
SEED_MAX_BUY_JIT = 16
# フィード小麦の補充は資金に余裕があるときだけ (元プレイヤーは d4-6 に小麦購入0)。
M5_FEED_FLOOR = True
FEED_CASH_FLOOR = 250
# M5b ステップ同期修正: ルートの steps[k].action は obs[k-1] に応答した行動
# (リプレイは事後状態を記録)。ローカルでは obs[k] に route[k+1] を提出しないと
# 農場が1ステップ遅れ、日境界直後の PICKUP (d7h1 COW3 など) が shed 非隣接で失敗し
# 動物が滞留する (位置比較で 0/240 一致することを実証済み)。
M5_OFFSET = True
# M5b HIRE 同期: ルートの HIRE オーダーを orig と同時刻に再現する (ハンドの
# スポーン位置を orig と一致させ、ハンド依存の農場作業のずれを防ぐ)。
# ギャップフィラーはルート雇用が完了した後 (hour > その日の最終 HIRE hour) のみ。
M5_ROUTE_HIRES = True
M5_GAP_HIRE_LATE = True
# M5e: ルートの BUY_ANIMAL を orig と同時刻に再現 (PICKUP 空振り防止)。
M5_ROUTE_ANIMALS = False
# M5f: ルートの小麦購入も orig と同時刻に再現 (給餌用小麦の枯渇防止)。
# FEED_CASH_FLOOR は資金難期に購入を止めすぎ、19頭の給餌需要に追いつかず
# GOOSE が脱走していた (orig は d7 に8個購入 vs 我々1個)。
M5_ROUTE_WHEAT = False
# M5h: 最終日 (d29) の小麦リザーブ解放 (当日給餌分のみ残して売却)
M5_ENDGAME_LIQUIDATE = True
# M5i: 価格暴落時は売却を控え、価格回復を待つ (実戦で WOOL が $1 まで暴落しても
# 売り続けていた。town 需要で $144 まで回復するため、保有が大幅有利)。
M5_PRICE_FLOOR = False
SELL_PRICE_FLOOR_FRAC = 0.5
# M6: 終盤イチゴ植え替え — ルートの PASS ターンを「その場の PLANT STRAWBERRY」に
# 静的に書き換える (移動なし=位置ドリフトなし。種は JIT が自動購入)。
M6_GAP_STRAWBERRY = False
M6_GAP_STRAWBERRY_DAYS = (14, 18)
M6_GAP_STRAWBERRY_MAX = 10
# M6b: ルートの d15-17 の WHEAT PLANT を STRAWBERRY に置換 (同一ユニット・同一位置・
# 同一タイミング)。ルート自身の水やり/施肥/収穫スケジュールがそのまま効くため
# 専任プランター方式 (M6) と違い枯死しない。終盤 (d24-28) のイチゴ生産を延長する
M6_SWAP_WHEAT = False
M6_SWAP_DAYS = (15, 18)
M6_SWAP_MAX = 12
# M5k: 肥料の購入を止める (最大のボトルネック)。動物19頭から毎日無料で
# 肥料が採れるのに、市場が「shed<2 なら常に買う」ため $100 で買って $60 で売る
# 往復損失を 454個/試合 ($45.4k) も続けていた。ルート自身の計画でも購入ゼロ。
M5_NO_FERT_BUY = True
# M6d: ギャップフィラーは農場が大きい時期 (d5-24) のみ雇用。
# 終盤 (d25-29) は作物が枯渇し水やり負荷が下がるため、fib(12)+fib(13)=$610/日の
# フィラー2人は過剰 (ルート生成・労働効率化の第一歩)
M6_GAP_WINDOW = True
M6_GAP_WINDOW_DAYS = (3, 22)

# 価格影響スコア用の市場パラメータ (kaggriculture.py の MARKET_PARAMS と同一)
_PRICE_PARAMS = {
    "WHEAT":      {"base": 25, "I0": 10000, "T": 400, "below_func": "sqrt", "below_target": 0.80, "above_func": "log", "above_target": 0.20},
    "CARROT":     {"base": 35, "I0": 10000, "T": 450, "below_func": "hinge", "below_target": 1.00, "above_func": "sqrt", "above_target": 0.70},
    "STRAWBERRY": {"base": 120, "I0": 10000, "T": 100, "below_func": "sqrt", "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": 10000, "T": 300, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.60},
    "EGG":        {"base": 50, "I0": 10000, "T": 332, "below_func": "hinge", "below_target": 0.40, "above_func": "log", "above_target": 0.20},
    "MILK":       {"base": 160, "I0": 10000, "T": 122, "below_func": "sqrt", "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": 10000, "T": 105, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": 10000, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}


def _price_shape(func, x, T=None):
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return x ** 0.5
    if func == "log":
        return math.log(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


def _market_price(item, inventory):
    p = _PRICE_PARAMS[item]
    base, I0, T = p["base"], p["I0"], p["T"]
    if inventory < I0:
        amp = p["below_target"] * base / _price_shape(p["below_func"], T)
        price = base + amp * _price_shape(p["below_func"], I0 - inventory)
    else:
        amp = p["above_target"] * base / _price_shape(p["above_func"], T)
        price = base - amp * _price_shape(p["above_func"], inventory - I0)
    return max(1, int(round(price)))


def _sell_impact_score(obs, order):
    """売却による価格下落 × 数量 (Kaito の _impact_score 移植)。"""
    if not (isinstance(order, (list, tuple)) and len(order) >= 3 and order[0] == "SELL"):
        return float("-inf")
    item = order[1]
    if item not in _PRICE_PARAMS:
        return float("-inf")
    try:
        qty = max(0, int(order[2]))
    except (TypeError, ValueError):
        return 0.0
    if qty <= 0:
        return 0.0
    market = obs.get("market", {}) or {}
    inv = market.get("inventory", {}) or {}
    prices = market.get("prices", {}) or {}
    current = int(inv.get(item, 10000) or 10000)
    quote = prices.get(item, _market_price(item, current))
    later = _market_price(item, current + qty)
    return float(qty) * max(0.0, quote - later)


def _rank_sell_orders(obs, market):
    """売却オーダーを価格影響の大きい順に並べ替える (同じスロット内で高額売却を先行)。"""
    rows = [
        (_sell_impact_score(obs, o), -idx, list(o))
        for idx, o in enumerate(market)
        if o and o[0] == "SELL"
    ]
    if len(rows) < 2:
        return market
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    ranked = iter(r[2] for r in rows)
    return [next(ranked) if o and o[0] == "SELL" else o for o in market]


def _fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _count_plant(entry, crop):
    n = 0
    acts = [entry.get("farmer") or ["PASS"]]
    acts += [h or ["PASS"] for h in entry.get("hands", [])]
    for a in acts:
        if a and len(a) > 1 and a[0] == "PLANT" and a[1] == crop:
            n += 1
    return n


def _plants_window(route, crop, W):
    """各 step から W ステップ以内の PLANT crop 回数 (ルート計画の種需要)。"""
    n = len(route)
    counts = [0] * n
    winsum = 0
    for i in range(min(W, n)):
        winsum += _count_plant(route[i], crop)
    counts[0] = winsum
    for s in range(1, n):
        winsum -= _count_plant(route[s - 1], crop)
        add = s + W - 1
        if add < n:
            winsum += _count_plant(route[add], crop)
        counts[s] = winsum
    return counts


def _next_plant_step(route, crop):
    """各 step 以降で最初に PLANT crop が現れる step (無ければ n)。"""
    n = len(route)
    out = [n] * n
    nxt = n
    for s in range(n - 1, -1, -1):
        if _count_plant(route[s], crop) > 0:
            nxt = s
        out[s] = nxt
    return out


def _animal_demand(route):
    """ルート農場の PICKUP <animal> から必要動物を導出 (品目ミスマッチ防止)。

    戻り値: {animal: {"target": 合計, "first_step": 最初のPICKUP step,
                       "last_step": 最後のPICKUP step}}
    """
    demand = {}
    for s, r in enumerate(route):
        acts = [r.get("farmer") or ["PASS"]]
        acts += [h or ["PASS"] for h in r.get("hands", [])]
        for act in acts:
            if act and len(act) >= 3 and act[0] == "PICKUP" and act[1] in ANIMAL_COST:
                d = demand.setdefault(act[1], {"target": 0, "first_step": s, "last_step": s})
                d["target"] += int(act[2])
                d["last_step"] = s
    return demand


_MOVE = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
_SHED_TILES = [(4, 4), (4, 5), (5, 4), (5, 5)]


def _sim_positions(route, hire_target):
    """ルートの農場アクションの実行位置をシミュレーションする (M3c)。

    ファーマーの位置は完全に決定論的。ハンドのスポーンは占有状況依存のため
    近似 (hire_target 人雇う前提)。戻り値:
      - build_pos: step -> その後の BUILD_PASTURE 位置の集合
      - plant_pos: step -> その後の PLANT 位置の集合
      - build_events: [(step, position)] のリスト
    """
    n = len(route)
    build_events = []
    plant_events = []
    farmer = [4, 4]
    hands = []
    for s, r in enumerate(route):
        hour = s % 24
        if hour == 0:
            farmer = [4, 4]
            hands = []
        n_hands = hire_target[s // 24]
        acts = [r.get("farmer") or ["PASS"]]
        acts += [h or ["PASS"] for h in r.get("hands", [])]

        def move(p, op):
            dx, dy = _MOVE[op]
            return [p[0] + dx, p[1] + dy]

        fa = acts[0]
        if fa[0] in _MOVE:
            farmer = move(farmer, fa[0])
        elif fa[0] == "BUILD_PASTURE":
            build_events.append((s, tuple(farmer)))
        elif fa[0] == "PLANT":
            plant_events.append((s, tuple(farmer)))
        for i in range(min(len(acts) - 1, n_hands)):
            ha = acts[i + 1]
            while len(hands) <= i:
                occ = {}
                for p in _SHED_TILES:
                    occ[p] = sum(1 for q in [farmer] + hands if tuple(q) == p)
                hands.append(list(min(occ, key=lambda p: (occ[p], _SHED_TILES.index(p)))))
            if ha[0] in _MOVE:
                hands[i] = move(hands[i], ha[0])
            elif ha[0] == "BUILD_PASTURE":
                build_events.append((s, tuple(hands[i])))
            elif ha[0] == "PLANT":
                plant_events.append((s, tuple(hands[i])))

    build_pos = [set() for _ in range(n)]
    plant_pos = [set() for _ in range(n)]
    for s in range(n):
        for e in build_events:
            if s <= e[0] < s + 48:
                build_pos[s].add(e[1])
        for e in plant_events:
            if s <= e[0] < s + 48:
                plant_pos[s].add(e[1])
    return build_pos, plant_pos, build_events


def _gap_strawberry_tile_ok(pos, s, plant_pos, build_pos):
    x, y = pos
    if not (0 <= x < 10 and 0 <= y < 10):
        return False
    if x in (4, 5) and y in (4, 5):
        return False
    if x >= 5 and y >= 5:
        return False
    if pos in plant_pos[s] or pos in build_pos[s]:
        return False
    return True


def _rewrite_gap_strawberries(route):
    """終盤の PASS を「その場の PLANT STRAWBERRY」に書き換える (M6)。"""
    if not M6_GAP_STRAWBERRY:
        return route
    d0, d1 = M6_GAP_STRAWBERRY_DAYS
    n = len(route)
    hire = [0] * 30
    for s, r in enumerate(route):
        for o in r.get("market", []):
            if o and o[0] == "HIRE":
                hire[s // 24] += 1
    build_pos, plant_pos, _ = _sim_positions(route, hire)
    new_route = [
        {
            "farmer": list(r.get("farmer") or ["PASS"]),
            "hands": [list(h) for h in r.get("hands", [])],
            "market": r.get("market", []),
        }
        for r in route
    ]
    farmer = [4, 4]
    hands = []
    count = 0
    for s in range(n):
        hour = s % 24
        day = s // 24
        if hour == 0:
            farmer = [4, 4]
            hands = []
        n_hands = hire[day]

        def move(p, op):
            dx, dy = _MOVE[op]
            return [p[0] + dx, p[1] + dy]

        fa = new_route[s]["farmer"]
        if d0 <= day < d1 and fa[0] == "PASS" and count < M6_GAP_STRAWBERRY_MAX:
            if _gap_strawberry_tile_ok(tuple(farmer), s, plant_pos, build_pos):
                new_route[s]["farmer"] = ["PLANT", "STRAWBERRY"]
                fa = new_route[s]["farmer"]
                count += 1
        if fa[0] in _MOVE:
            farmer = move(farmer, fa[0])
        acts = new_route[s]["hands"]
        for i in range(min(len(acts), n_hands)):
            while len(hands) <= i:
                occ = {}
                for p in _SHED_TILES:
                    occ[p] = sum(1 for q in [farmer] + hands if tuple(q) == p)
                hands.append(list(min(occ, key=lambda p: (occ[p], _SHED_TILES.index(p)))))
            ha = acts[i]
            if d0 <= day < d1 and ha[0] == "PASS" and count < M6_GAP_STRAWBERRY_MAX:
                if _gap_strawberry_tile_ok(tuple(hands[i]), s, plant_pos, build_pos):
                    new_route[s]["hands"][i] = ["PLANT", "STRAWBERRY"]
                    ha = new_route[s]["hands"][i]
                    count += 1
            if ha[0] in _MOVE:
                hands[i] = move(hands[i], ha[0])
    return new_route


def _rewrite_swap_wheat_strawberry(route):
    """ルートの d15-17 の WHEAT PLANT を STRAWBERRY に置換 (M6b)。

    同一ユニット・同一位置・同一ステップで作物だけ入れ替えるため、位置ドリフト
    ゼロ。置換後のタイルはルート自身の水やりスケジュールで維持される。
    """
    if not M6_SWAP_WHEAT:
        return route
    d0, d1 = M6_SWAP_DAYS
    new_route = [
        {
            "farmer": list(r.get("farmer") or ["PASS"]),
            "hands": [list(h) for h in r.get("hands", [])],
            "market": r.get("market", []),
        }
        for r in route
    ]
    count = 0
    for s, r in enumerate(new_route):
        day = s // 24
        if not (d0 <= day < d1):
            continue
        fa = r["farmer"]
        if count < M6_SWAP_MAX and fa[0] == "PLANT" and fa[1] == "WHEAT":
            r["farmer"] = ["PLANT", "STRAWBERRY"]
            count += 1
        for i, h in enumerate(r["hands"]):
            if count < M6_SWAP_MAX and h[0] == "PLANT" and h[1] == "WHEAT":
                r["hands"][i] = ["PLANT", "STRAWBERRY"]
                count += 1
    return new_route


def build_plan(route, lookahead=LOOKAHEAD):
    n = len(route)
    market_buys = {}
    animal_pace = {}  # animal -> day -> 購入ユニット数 (ルートのペース)
    for s, r in enumerate(route):
        day = s // 24
        for o in r.get("market", []):
            if o and o[0] == "BUY_ANIMAL" and len(o) >= 3 and o[1] in ANIMAL_COST:
                a = o[1]
                market_buys[a] = market_buys.get(a, 0) + int(o[2])
                pace = animal_pace.setdefault(a, {})
                pace[day] = pace.get(day, 0) + int(o[2])
    animals = {}
    for a, d in _animal_demand(route).items():
        # 数量は市場購入実績を上限 (PICKUP は失敗分を再試行するため過大に出る)
        animals[a] = {
            "target": min(d["target"], market_buys.get(a, 0)),
            "first_step": d["first_step"],
            "last_step": d["last_step"],
        }
    # 1日あたり購入数の累積 (資金が許せばルートと同じ日に、許さなければ後日キャッチアップ)
    animal_cum = {}
    for a, pace in animal_pace.items():
        cum = []
        total = 0
        for day in range(30):
            total += pace.get(day, 0)
            cum.append(total)
        animal_cum[a] = cum
    plan = {
        "plants": {c: _plants_window(route, c, lookahead) for c in CROPS4},
        "next_plant": {c: _next_plant_step(route, c) for c in CROPS4},
        "animals": animals,
        "animal_cum": animal_cum,
        "animal_total": sum(d["target"] for d in animals.values()),
        "seed_first_day": {c: None for c in CROPS4},
        "land_total": 0,
        "first_land_day": None,
        "hire_target": [0] * 30,
        "uses_fertilizer": False,
    }
    for s, r in enumerate(route):
        day = s // 24
        for o in r.get("market", []):
            if not o:
                continue
            op = o[0]
            if op == "HIRE":
                plan["hire_target"][day] += 1
            elif op == "BUY_LAND":
                plan["land_total"] += 1
                if plan["first_land_day"] is None:
                    plan["first_land_day"] = day
            elif op == "BUY_SEED" and o[1] in CROPS4 and plan["seed_first_day"][o[1]] is None:
                plan["seed_first_day"][o[1]] = day
    # 一度も買われない作物は 99 (事実上無限) にする — None のままだと
    # 市場ロジックの `day < seed_first_day` 比較が TypeError になり agent 全体が
    # PASS フォールバックする (E018 強敵評価で Rocket Zech 等のルートが崩壊した原因)
    for c in CROPS4:
        if plan["seed_first_day"][c] is None:
            plan["seed_first_day"][c] = 99
    if M5_JIT_SEEDS:
        plan["plants_jit"] = {c: _plants_window(route, c, SEED_LOOKAHEAD_JIT) for c in CROPS4}
    plan["jit_seeds"] = M5_JIT_SEEDS
    plan["feed_floor"] = M5_FEED_FLOOR
    plan["feed_cash_floor"] = FEED_CASH_FLOOR
    plan["offset"] = M5_OFFSET
    plan["route_hires"] = M5_ROUTE_HIRES
    plan["gap_late"] = M5_GAP_HIRE_LATE
    plan["gap_max"] = GAP_HIRE_MAX
    plan["route_animals"] = M5_ROUTE_ANIMALS
    plan["n"] = n
    # M5b: ルートの HIRE オーダーのステップ分布 (orig と同時刻に雇用するため)
    hire_orders = [0] * n
    for s, r in enumerate(route):
        for o in r.get("market", []):
            if o and o[0] == "HIRE":
                hire_orders[s] += 1
    hire_cum = [0] * n
    c = 0
    for s in range(n):
        c += hire_orders[s]
        hire_cum[s] = c
    plan["hire_orders"] = hire_orders
    plan["hire_cum"] = hire_cum
    plan["gap_hour"] = [3] * 30
    for s, r in enumerate(route):
        for o in r.get("market", []):
            if o and o[0] == "HIRE":
                plan["gap_hour"][s // 24] = s % 24
    # M5e: ルートの BUY_ANIMAL オーダーのステップ分布 (orig と同じタイミングで
    # 購入し、PICKUP に間に合わせる。first_step 順のペース購入では品目間の
    # 優先が orig と逆転して PICKUP が空振りする — GOOSE の d6h3 PICKUP 等)
    animal_buy_steps = {}
    for s, r in enumerate(route):
        for o in r.get("market", []):
            if o and o[0] == "BUY_ANIMAL" and o[1] in ANIMAL_COST:
                animal_buy_steps.setdefault(s, []).append((o[1], int(o[2]) if len(o) > 2 else 1))
    plan["animal_buy_steps"] = animal_buy_steps
    # M5f: ルートの小麦購入 (BUY_PRODUCT WHEAT) のステップ分布
    wheat_buy_steps = {}
    for s, r in enumerate(route):
        for o in r.get("market", []):
            if o and o[0] == "BUY_PRODUCT" and o[1] == "WHEAT":
                wheat_buy_steps[s] = wheat_buy_steps.get(s, 0) + (int(o[2]) if len(o) > 2 else 1)
    plan["wheat_buy_steps"] = wheat_buy_steps
    plan["route_wheat"] = M5_ROUTE_WHEAT
    plan["endgame_liquidate"] = M5_ENDGAME_LIQUIDATE
    plan["price_floor"] = M5_PRICE_FLOOR
    plan["m6_gap"] = M6_GAP_STRAWBERRY
    plan["m6_swap"] = M6_SWAP_WHEAT
    plan["no_fert_buy"] = M5_NO_FERT_BUY
    plan["gap_window"] = M6_GAP_WINDOW
    plan["gap_window_days"] = M6_GAP_WINDOW_DAYS
    build_pos, plant_pos, build_events = _sim_positions(route, plan["hire_target"])
    plan["build_pos"] = build_pos
    plan["plant_pos"] = plant_pos
    plan["build_events"] = build_events
    pasture_steps = {}
    for s, pos in build_events:
        pasture_steps.setdefault(pos, s)
    plan["pasture_steps"] = pasture_steps
    plan["animal_ready_day"] = max(
        (plan["animals"][a]["last_step"] // 24 for a in plan["animals"]), default=99
    ) + 1
    for s, r in enumerate(route):
        for act in [r.get("farmer") or ["PASS"]] + [h or ["PASS"] for h in r.get("hands", [])]:
            if act and act[0] == "FERTILIZE":
                plan["uses_fertilizer"] = True
            if act and len(act) >= 3 and act[0] == "PICKUP" and act[1] == "FERTILIZER":
                plan["uses_fertilizer"] = True
        for o in r.get("market", []):
            if o and o[0] == "BUY_PRODUCT" and o[1] == "FERTILIZER":
                plan["uses_fertilizer"] = True
    return plan


def _is_weed(tile):
    return isinstance(tile, dict) and tile.get("kind") == "WEED"


def _step_toward(fx, fy, tx, ty):
    if fx > tx:
        return "WEST"
    if fx < tx:
        return "EAST"
    if fy > ty:
        return "NORTH"
    if fy < ty:
        return "SOUTH"
    return None


def _inventory_of(private, farm, pos):
    """ユニットの手持ち (ファーマー=0、ハンド=i+1)。"""
    idx = 0
    if tuple(pos) != tuple(farm["farmer"]):
        for i, h in enumerate(farm["hands"]):
            if tuple(h) == tuple(pos):
                idx = i + 1
                break
    invs = private.get("inventories", [])
    return invs[idx] if idx < len(invs) else {}


def _count_pastures(tiles):
    return sum(1 for row in tiles for t in row if isinstance(t, dict) and t.get("kind") == "PASTURE")


def _empty_pasture_pos(tiles):
    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            if isinstance(t, dict) and t.get("kind") == "PASTURE" and "animal" not in t:
                return (x, y)
    return None


def _shed_adjacent(pos):
    x, y = pos
    return x in (4, 5) and y in (4, 5)


def _reactive_animal_action(obs, farm, private, pos, plan):
    """ギャップフィラー (動物ロール): 給餌・配置をリアクティブに補完する。

    ルートの PICKUP ステップが資金遅延で失敗した場合、動物が shed に残り
    ミルク/羊毛を生まない。また、ルートの給餌スケジュールは自分の配置
    分しか想定しておらず、リアクティブ配置した動物は飢えて脱走する。
    処理順: (1) 未給餌動物への給餌 (脱走防止) (2) 滞留動物の配置 (3) 牧場建設
    """
    tiles = farm["tiles"]
    board = len(tiles)
    x, y = pos
    inv = _inventory_of(private, farm, pos)
    shed = private.get("shed", {}) or {}
    animal_total = plan.get("animal_total", 0)
    placed = sum(1 for row in tiles for t in row if isinstance(t, dict) and "animal" in t)
    carrying = [a for a in ("COW", "SHEEP", "GOOSE") if inv.get(a, 0) > 0]
    shed_animals = {a: shed.get(a, 0) for a in ("COW", "SHEEP", "GOOSE") if shed.get(a, 0) > 0}

    # --- (1) 手持ちの動物を空き牧場へ配置 (最優先: 持ったまま放置すると
    # 夜に shed へ戻されて配置が永遠に進まない) ---
    if carrying:
        a = carrying[0]
        ep = _empty_pasture_pos(tiles)
        if ep == pos:
            return ["PLACE", a, 1]
        if ep:
            step = _step_toward(x, y, ep[0], ep[1])
            if step:
                return [step]
        # 空き牧場が無ければ一旦 PASS (建設はルートの作物計画を圧迫するためしない。
        # ルート自身の牧場建設が後から配置機会を作る)
        return ["PASS"]

    # --- (2) 未給餌の配置動物への給餌 (脱走防止) ---
    unfed = [
        (tx, ty) for ty in range(board) for tx in range(board)
        if isinstance(tiles[ty][tx], dict) and "animal" in tiles[ty][tx]
        and not tiles[ty][tx].get("fed_today", False)
    ]
    if unfed:
        if inv.get("WHEAT", 0) > 0:
            ux, uy = min(unfed, key=lambda p: abs(x - p[0]) + abs(y - p[1]))
            if (ux, uy) == (x, y):
                return ["FEED"]
            step = _step_toward(x, y, ux, uy)
            if step:
                return [step]
        elif shed.get("WHEAT", 0) > 0:
            if _shed_adjacent(pos):
                return ["PICKUP", "WHEAT", 2]
            step = _step_toward(x, y, 4, 4)
            if step:
                return [step]

    # --- (3) 滞留動物のピックアップ (空き牧場 or 建設余地があるとき) ---
    if shed_animals:
        can_place = _empty_pasture_pos(tiles) is not None or _count_pastures(tiles) < max(animal_total, placed + sum(shed_animals.values()))
        if can_place:
            if _shed_adjacent(pos):
                a = max(shed_animals, key=lambda k: shed_animals[k])
                return ["PICKUP", a, 1]
            step = _step_toward(x, y, 4, 4)
            if step:
                return [step]

    # --- (4) 空きタイルに牧場を建設 (配置余地の確保) ---
    if shed_animals and _count_pastures(tiles) < placed + sum(shed_animals.values()):
        for ty in range(board):
            for tx in range(board):
                if tiles[ty][tx] is None:
                    if (tx, ty) == (x, y):
                        return ["BUILD_PASTURE"]
                    step = _step_toward(x, y, tx, ty)
                    if step:
                        return [step]
    return ["PASS"]


def _reactive_animal_action(obs, farm, private, pos, plan, day):
    """ギャップフィラー (動物専任・ルートの最終 PICKUP 後のみ使用)。

    滞留動物の配置と給餌を担当する。動物は2日連続の未給餌で脱走するため
    給餌を最優先する。
    """
    tiles = farm["tiles"]
    board = len(tiles)
    x, y = pos
    inv = _inventory_of(private, farm, pos)
    shed = private.get("shed", {}) or {}
    placed = sum(1 for row in tiles for t in row if isinstance(t, dict) and "animal" in t)
    carrying = [a for a in ("COW", "SHEEP", "GOOSE") if inv.get(a, 0) > 0]
    shed_animals = {a: shed.get(a, 0) for a in ("COW", "SHEEP", "GOOSE") if shed.get(a, 0) > 0}

    # 脱走リスクの高い動物 (1日以上未給餌) を最優先で給餌
    at_risk = [
        (tx, ty) for ty in range(board) for tx in range(board)
        if isinstance(tiles[ty][tx], dict) and "animal" in tiles[ty][tx]
        and not tiles[ty][tx].get("fed_today", False)
        and tiles[ty][tx].get("consecutive_unfed", 0) >= 1
    ]
    if at_risk:
        if inv.get("WHEAT", 0) > 0:
            ux, uy = min(at_risk, key=lambda p2: abs(x - p2[0]) + abs(y - p2[1]))
            if (ux, uy) == (x, y):
                return ["FEED"]
            step = _step_toward(x, y, ux, uy)
            if step:
                return [step]
        elif shed.get("WHEAT", 0) > 0:
            if _shed_adjacent(pos):
                return ["PICKUP", "WHEAT", 3]
            step = _step_toward(x, y, 4, 4)
            if step:
                return [step]

    # 手持ちの動物を配置 (空き牧場 → ルート意図位置に建設)
    if carrying:
        a = carrying[0]
        ep = _empty_pasture_pos(tiles)
        if ep == pos:
            return ["PLACE", a, 1]
        if ep:
            step = _step_toward(x, y, ep[0], ep[1])
            if step:
                return [step]
        plan_pos = None
        for (px, py), bstep in plan.get("pasture_steps", {}).items():
            if day * 24 <= bstep:
                continue
            t = tiles[py][px]
            if t is None or (isinstance(t, dict) and t.get("kind") == "WEED"):
                d = abs(x - px) + abs(y - py)
                if plan_pos is None or d < plan_pos[0]:
                    plan_pos = (d, px, py, t)
        if plan_pos:
            _, px, py, t = plan_pos
            if (px, py) == (x, y):
                return ["DIG"] if (isinstance(t, dict) and t.get("kind") == "WEED") else ["BUILD_PASTURE"]
            step = _step_toward(x, y, px, py)
            if step:
                return [step]
        return ["PASS"]

    # ピックアップ (空き牧場 or 建設余地あり)
    if shed_animals:
        can_place = _empty_pasture_pos(tiles) is not None or any(
            tiles[py][px] is None or (isinstance(tiles[py][px], dict) and tiles[py][px].get("kind") == "WEED")
            for (px, py) in plan.get("pasture_steps", {})
            if day * 24 > plan["pasture_steps"][(px, py)]
        )
        if can_place:
            if _shed_adjacent(pos):
                a = max(shed_animals, key=lambda k: shed_animals[k])
                return ["PICKUP", a, 1]
            step = _step_toward(x, y, 4, 4)
            if step:
                return [step]

    # 通常の未給餌動物も給餌 (配置済み動物の生存維持)
    unfed = [
        (tx, ty) for ty in range(board) for tx in range(board)
        if isinstance(tiles[ty][tx], dict) and "animal" in tiles[ty][tx]
        and not tiles[ty][tx].get("fed_today", False)
    ]
    if unfed:
        if inv.get("WHEAT", 0) > 0:
            ux, uy = min(unfed, key=lambda p2: abs(x - p2[0]) + abs(y - p2[1]))
            if (ux, uy) == (x, y):
                return ["FEED"]
            step = _step_toward(x, y, ux, uy)
            if step:
                return [step]
        elif shed.get("WHEAT", 0) > 0:
            if _shed_adjacent(pos):
                return ["PICKUP", "WHEAT", 3]
            step = _step_toward(x, y, 4, 4)
            if step:
                return [step]
    return ["PASS"]


def _reactive_hand_action(obs, farm, private, pos, day, plan=None, step=0, is_planter=False):
    """ギャップフィラーハンド: 水やり・雑草除去を最優先し、手が空いたら
    滞留動物の配置を補完する (ルートの PICKUP 完了後のみ — 干渉防止)。

    ルートのハンドが位置ドリフトで水やりを漏らした作物の枯死を防ぐのが
    主目的。M3c: ルートが今後48ステップ以内に BUILD_PASTURE/PLANT する
    位置の雑草を優先除去し、牧場建設・植え付けの失敗を防ぐ。
    """
    tiles = farm["tiles"]
    board = len(tiles)
    x, y = pos
    tile = tiles[y][x]
    inv = _inventory_of(private, farm, pos)
    # M6: 植え付け専任ハンド — 水やりより植えを優先 (最後のギャップハンドのみ)。
    if is_planter and plan and plan.get("m6_gap") and M6_GAP_STRAWBERRY_DAYS[0] <= day < M6_GAP_STRAWBERRY_DAYS[1] \
            and _m6_gap_count(farm) < M6_GAP_STRAWBERRY_MAX:
        # (1) ギャップイチゴの水やり最優先 (植えた当日に水をやらないと枯死する)
        need = [(tx, ty) for ty in range(board) for tx in range(board)
                if isinstance(tiles[ty][tx], dict)
                and tiles[ty][tx].get("kind") == "PLANT"
                and tiles[ty][tx].get("crop") == "STRAWBERRY"
                and tiles[ty][tx].get("planted_day", 0) >= 14
                and not tiles[ty][tx].get("watered_today")]
        if need:
            ux, uy = min(need, key=lambda p2: abs(x - p2[0]) + abs(y - p2[1]))
            if (ux, uy) == (x, y):
                return ["WATER"]
            step_ = _step_toward(x, y, ux, uy)
            if step_:
                return [step_]
        # (2) 新規植え
        best = None
        for ty in range(board):
            for tx in range(board):
                t = tiles[ty][tx]
                if t is not None:
                    continue
                if tx in (4, 5) and ty in (4, 5):
                    continue
                if tx >= 5 and ty >= 5:
                    continue
                if plan and step < len(plan.get("plant_pos", [])):
                    if (tx, ty) in plan["plant_pos"][step] or (tx, ty) in plan["build_pos"][step]:
                        continue
                d2 = abs(x - tx) + abs(y - ty)
                if best is None or d2 < best[0]:
                    best = (d2, tx, ty)
        if best:
            _, tx, ty = best
            if (tx, ty) == (x, y):
                return ["PLANT", "STRAWBERRY"]
            step_ = _step_toward(x, y, tx, ty)
            if step_:
                return [step_]
    urgent = set()
    if plan and step < len(plan.get("build_pos", [])):
        urgent = plan["build_pos"][step] | plan["plant_pos"][step]

    # --- 現在地タイルの処理 (水やり > 雑草 > 作物収穫) ---
    if isinstance(tile, dict):
        if tile.get("kind") == "PLANT" and not tile.get("watered_today"):
            return ["WATER"]
        if tile.get("kind") == "WEED":
            return ["DIG"]
        if tile.get("kind") == "PLANT":
            crop = tile.get("crop")
            age = day - tile.get("planted_day", 0)
            if crop in ("STRAWBERRY", "WHEAT", "CARROT") and tile.get("yield_units", 0) > 0:
                fy = {"STRAWBERRY": 10, "WHEAT": 4, "CARROT": 3}[crop]
                if age >= fy:
                    return ["HARVEST"]
    best = None  # (dist, kind, urgent, tx, ty)
    for ty in range(board):
        for tx in range(board):
            t = tiles[ty][tx]
            if not isinstance(t, dict):
                continue
            if t.get("kind") == "PLANT" and not t.get("watered_today"):
                key = (abs(x - tx) + abs(y - ty), 0, 0, tx, ty)
            elif t.get("kind") == "WEED":
                key = (abs(x - tx) + abs(y - ty), 1, 0 if (tx, ty) in urgent else 1, tx, ty)
            else:
                continue
            if best is None or key < best:
                best = key
    if best:
        step = _step_toward(x, y, best[3], best[4])
        if step:
            return [step]

    # --- M6: ギャップイチゴ植え (水やり・雑草の次に優先。最寄りの空きタイルへ) ---
    # リアクティブハンドは自由移動のため、ルートの位置スケジュールを壊さない。
    if (plan and plan.get("m6_gap") and M6_GAP_STRAWBERRY_DAYS[0] <= day < M6_GAP_STRAWBERRY_DAYS[1]
            and _m6_gap_count(farm) < M6_GAP_STRAWBERRY_MAX):
        best = None
        for ty in range(board):
            for tx in range(board):
                t = tiles[ty][tx]
                if t is not None:
                    continue
                if tx in (4, 5) and ty in (4, 5):
                    continue
                if tx >= 5 and ty >= 5:
                    continue
                if plan and step < len(plan.get("plant_pos", [])):
                    if (tx, ty) in plan["plant_pos"][step] or (tx, ty) in plan["build_pos"][step]:
                        continue
                d2 = abs(x - tx) + abs(y - ty)
                if best is None or d2 < best[0]:
                    best = (d2, tx, ty)
        if best:
            _, tx, ty = best
            if (tx, ty) == (x, y):
                return ["PLANT", "STRAWBERRY"]
            step_ = _step_toward(x, y, tx, ty)
            if step_:
                return [step_]

    # --- 収穫 (水やり/雑草が済んだら。メロンはルートのタイミングに任せる) ---
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop = tile.get("crop")
        age = day - tile.get("planted_day", 0)
        if crop in ("STRAWBERRY", "WHEAT", "CARROT") and tile.get("yield_units", 0) > 0:
            fy = {"STRAWBERRY": 10, "WHEAT": 4, "CARROT": 3}[crop]
            if age >= fy:
                return ["HARVEST"]
    for ty in range(board):
        for tx in range(board):
            t = tiles[ty][tx]
            if not (isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("yield_units", 0) > 0):
                continue
            crop = t.get("crop")
            if crop not in ("STRAWBERRY", "WHEAT", "CARROT"):
                continue
            fy = {"STRAWBERRY": 10, "WHEAT": 4, "CARROT": 3}[crop]
            if day - t.get("planted_day", 0) >= fy:
                step = _step_toward(x, y, tx, ty)
                if step:
                    return [step]
                return ["HARVEST"]

    # --- 動物タスク (ルートの最終 PICKUP を過ぎてから。牧場はルート建設分のみ) ---
    if plan and plan.get("animals"):
        shed = private.get("shed", {}) or {}
        carrying = [a for a in ("COW", "SHEEP", "GOOSE") if inv.get(a, 0) > 0]
        shed_animals = {a: shed.get(a, 0) for a in ("COW", "SHEEP", "GOOSE") if shed.get(a, 0) > 0}
        last_pickup_day = max(
            (plan["animals"][a]["last_step"] // 24 for a in plan["animals"]), default=-1
        )
        if day > last_pickup_day:
            if carrying:
                a = carrying[0]
                ep = _empty_pasture_pos(tiles)
                if ep == pos:
                    return ["PLACE", a, 1]
                if ep:
                    step = _step_toward(x, y, ep[0], ep[1])
                    if step:
                        return [step]
                # 空き牧場が無ければルートの意図した牧場位置に建設
                # (建設ステップを過ぎて、タイルが空/雑草の場合のみ)
                plan_pos = None
                for (px, py), bstep in plan.get("pasture_steps", {}).items():
                    if step <= bstep:
                        continue
                    t = tiles[py][px]
                    if t is None or (isinstance(t, dict) and t.get("kind") == "WEED"):
                        d = abs(x - px) + abs(y - py)
                        if plan_pos is None or d < plan_pos[0]:
                            plan_pos = (d, px, py, t)
                if plan_pos:
                    _, px, py, t = plan_pos
                    if (px, py) == (x, y):
                        return ["DIG"] if (isinstance(t, dict) and t.get("kind") == "WEED") else ["BUILD_PASTURE"]
                    step_ = _step_toward(x, y, px, py)
                    if step_:
                        return [step_]
                return ["PASS"]
            if shed_animals and _empty_pasture_pos(tiles) is not None:
                if _shed_adjacent(pos):
                    a = max(shed_animals, key=lambda k: shed_animals[k])
                    return ["PICKUP", a, 1]
                step = _step_toward(x, y, 4, 4)
                if step:
                    return [step]
    return ["PASS"]


def _repair(action, pos, tiles):
    if not action:
        return ["PASS"]
    op = action[0]
    fx, fy = pos
    tile = tiles[fy][fx]
    if op in ("PLANT", "BUILD_COOP", "BUILD_PASTURE") and _is_weed(tile):
        return ["DIG"]
    return action


def _count_placed_animals(tiles):
    cnt = {}
    for row in tiles:
        for t in row:
            if isinstance(t, dict) and "animal" in t:
                cnt[t["animal"]] = cnt.get(t["animal"], 0) + 1
    return cnt


def _total_animals(shed, invs, placed):
    n = 0
    for a in ("GOOSE", "COW", "SHEEP"):
        n += shed.get(a, 0)
        n += sum(inv.get(a, 0) for inv in invs)
    n += sum(placed.values())
    return n


def _reactive_market(obs, farm, private, plan, step, day, hour):
    prices = (obs.get("market", {}) or {}).get("prices", {})
    shed = private.get("shed", {}) or {}
    seeds = private.get("seeds", {}) or {}
    invs = private.get("inventories", []) or []
    money = farm.get("money", 0)
    placed = _count_placed_animals(farm["tiles"])
    n_animals = _total_animals(shed, invs, placed)
    market = []

    # --- 売却 (最優先) ---
    # M5i: 暴落中の商品は保有して回復を待つ (最終日は強制売却)。
    for item in SELL_ORDER:
        n = shed.get(item, 0)
        if n <= 0:
            continue
        if plan.get("price_floor") and day < 29 and item in _PRICE_PARAMS:
            base = _PRICE_PARAMS[item]["base"]
            if prices.get(item, base) < SELL_PRICE_FLOOR_FRAC * base:
                continue
        if item == "WHEAT":
            # ルートのユニットが毎日10-36個を PICKUP するため、shed 在庫を
            # 残して売る (実在動物数のフィードも確保)。売りすぎない範囲で
            # d5-9 の資金源にする (雇用が止まると水やり崩壊で作物が枯れる)
            # M5h: 最終日 (d29) は当日分の給餌だけ残して全て売却する。
            # 未給餌は脱走 (2日連続) せず基本生産も入るため、CARE ボーナスと
            # 当日卵の分だけ残せばよい (従来は 38 個が死蔵されていた = ~$1.5k)
            if plan.get("endgame_liquidate") and day >= 29:
                reserve = sum(placed.values())
            else:
                reserve = max(6, n_animals * FEED_RESERVE_PER_ANIMAL)
            n = max(0, n - reserve)
            if n <= 0:
                continue
        market.append(["SELL", item, min(SELL_CHUNK[item], n)])

    # --- 雇用 (hour1-3 が上位のパターン。ルートの1日当たり雇用数が上限。
    # hour1 で資金が足りなくても hour2 で再試行される。
    # M3: ルート目標に達したらギャップフィラーを最大2人追加) ---
    # M5b: ルートの HIRE オーダーを orig と同じステップで再現する
    # (ハンドのスポーン位置一致)。ギャップフィラーはルートの当日分の雇用が
    # 完了する hour を過ぎてから。
    if plan.get("route_hires") and "hire_cum" in plan:
        n_route = len(plan["hire_cum"])
        rs = min(step + 1, n_route - 1)
        day_start = day * 24
        cum_before = plan["hire_cum"][day_start - 1] if day_start > 0 else 0
        target = plan["hire_cum"][rs] - cum_before
        n_today = farm.get("hires_today", 0)
        while n_today < target:
            cost = _fib(n_today)
            if money < cost:
                break
            market.append(["HIRE"])
            n_today += 1
            money -= cost
        if (plan.get("gap_late") and hour > plan["gap_hour"][day]
                and (not plan.get("gap_window")
                     or plan["gap_window_days"][0] <= day <= plan["gap_window_days"][1])):
            cap = plan["hire_target"][day] + plan["gap_max"]
            if plan.get("m6_gap") and M6_GAP_STRAWBERRY_DAYS[0] <= day < M6_GAP_STRAWBERRY_DAYS[1]:
                cap += 1
            while n_today < cap:
                cost = _fib(n_today)
                if money < cost:
                    break
                market.append(["HIRE"])
                n_today += 1
                money -= cost
    elif hour in (1, 2, 3) and day < len(plan["hire_target"]):
        n_today = farm.get("hires_today", 0)
        cap = plan["hire_target"][day] - n_today + plan["gap_max"]
        while cap > 0 and n_today < plan["hire_target"][day] + plan["gap_max"]:
            cost = _fib(n_today)
            if money < cost:
                break
            market.append(["HIRE"])
            n_today += 1
            cap -= 1
            money -= cost

    # --- フィード用小麦の購入 (shed 在庫ターゲット管理: ルートの PICKUP 需要を満たす。
    # ルートは一日中 PICKUP するため常時補充が必要) ---)
    # M5: 資金フロアを下回る間は購入を止める (収穫小麦に任せる)。元プレイヤーは
    # d4-6 の資金難期に小麦購入0で、購入は資金に余裕がある日に集中している。
    shed_wheat = shed.get("WHEAT", 0)
    if plan.get("route_wheat") and "wheat_buy_steps" in plan:
        n = plan.get("n", len(plan["plants"]["WHEAT"]))
        rs = min(step + 1, n - 1)
        qty = plan["wheat_buy_steps"].get(rs, 0)
        if qty > 0:
            wheat_price = prices.get("WHEAT", 25)
            buy = min(qty, int(money) // max(1, wheat_price))
            if buy > 0:
                market.append(["BUY_PRODUCT", "WHEAT", buy])
                money -= buy * wheat_price
    if shed_wheat < WHEAT_STOCK_TARGET and (not plan.get("feed_floor") or money >= plan["feed_cash_floor"]):
        wheat_price = prices.get("WHEAT", 25)
        buy = min(WHEAT_STOCK_TARGET - shed_wheat, 4, int(money) // max(1, wheat_price))
        if buy > 0:
            market.append(["BUY_PRODUCT", "WHEAT", buy])
            money -= buy * wheat_price

    # --- 動物購入 (ルート農場が必要とする品目。累積ペースで購入、種より優先:
    # 動物は日次ゲートで逃すと配置機会が失われる (E018-M3c の検証) ---)
    # M5e: ルートの BUY_ANIMAL オーダーを orig と同じステップで再現する
    # (first_step 順のペース購入は品目間の優先が orig と逆転し、GOOSE の
    # d6h3 PICKUP 等が空振りして配置が 18/18 → 15/18 に落ちていた)。
    # キャッチアップ (資金不足で逃した分) は hour>=12 に当日分のみ。
    if plan["animals"]:
        n = plan.get("n", len(plan["plants"]["WHEAT"]))
        if plan.get("route_animals") and "animal_buy_steps" in plan:
            owned = {a: shed.get(a, 0) + placed.get(a, 0) + sum(inv.get(a, 0) for inv in invs)
                     for a in plan["animals"]}
            rs = min(step + 1, n - 1)
            for a, qty in plan["animal_buy_steps"].get(rs, []):
                d = plan["animals"][a]
                while qty > 0 and owned.get(a, 0) < d["target"] and money >= ANIMAL_COST[a] + ANIMAL_BUFFER:
                    market.append(["BUY_ANIMAL", a, 1])
                    owned[a] = owned.get(a, 0) + 1
                    qty -= 1
                    money -= ANIMAL_COST[a]
            if hour >= 12:
                for a in sorted(plan["animals"], key=lambda x: plan["animals"][x]["first_step"]):
                    d = plan["animals"][a]
                    if day < d["first_step"] // 24:
                        continue
                    need_by_today = plan["animal_cum"][a][day] if day < 30 else d["target"]
                    o = owned.get(a, 0)
                    while o < min(need_by_today, d["target"]) and money >= ANIMAL_COST[a] + ANIMAL_BUFFER:
                        market.append(["BUY_ANIMAL", a, 1])
                        o += 1
                        money -= ANIMAL_COST[a]
        else:
            for a in sorted(plan["animals"], key=lambda x: plan["animals"][x]["first_step"]):
                d = plan["animals"][a]
                if day < d["first_step"] // 24:
                    continue
                need_by_today = plan["animal_cum"][a][day] if day < 30 else d["target"]
                owned = shed.get(a, 0) + placed.get(a, 0) + sum(inv.get(a, 0) for inv in invs)
                while owned < min(need_by_today, d["target"]) and money >= ANIMAL_COST[a] + ANIMAL_BUFFER:
                    market.append(["BUY_ANIMAL", a, 1])
                    owned += 1
                    money -= ANIMAL_COST[a]

    # --- 種の補充 (hour2 以降: hour0-1 の種購入が雇用資金を食い潰し、
    # ルートと雇用数がずれてハンドのスポーン位置がドリフトするのを防ぐ) ---
    # 購入日はルートの最初の種購入日以降。植え付けが近い作物から優先。
    # M5 JIT: 需要ウィンドウを「次の SEED_LOOKAHEAD_JIT ステップ」に縮小。
    # 種は今ステップのユニット行動後に購入されるため、s+1 以降の植え付けに
    # 間に合えばよい (大量前買いで d0-9 の資金を枯渇させない)。
    if hour >= 2 and step < len(plan["plants"]["WHEAT"]):
        n = len(plan["plants"]["WHEAT"])
        for crop in sorted(CROPS4, key=lambda c: (plan["next_plant"][c][step], c)):
            if day < plan["seed_first_day"][crop]:
                continue
            if plan.get("jit_seeds"):
                off = 2 if plan.get("offset") else 1
                need = plan["plants_jit"][crop][min(step + off, n - 1)] + SEED_BUFFER
                max_buy = SEED_MAX_BUY_JIT
            else:
                need = plan["plants"][crop][step] + SEED_BUFFER
                max_buy = SEED_MAX_BUY
            have = seeds.get(crop, 0)
            if have >= need or money < SEED_COST[crop]:
                continue
            buy = min(need - have, max_buy, int(money) // SEED_COST[crop])
            if buy > 0:
                market.append(["BUY_SEED", crop, buy])
                money -= buy * SEED_COST[crop]

    # --- M6: ギャップイチゴの種の補充 (窓内で種2個未満なら1個買う) ---
    if (plan.get("m6_gap") and M6_GAP_STRAWBERRY_DAYS[0] <= day < M6_GAP_STRAWBERRY_DAYS[1]
            and seeds.get("STRAWBERRY", 0) < 2
            and _m6_gap_count(farm) < M6_GAP_STRAWBERRY_MAX
            and money >= SEED_COST["STRAWBERRY"] + 50):
        market.append(["BUY_SEED", "STRAWBERRY", 1])
        money -= SEED_COST["STRAWBERRY"]

    # --- 土地購入 (day5+ のタイミング、資金ゲート付き) ---
    if plan["land_total"] > 0 and plan["first_land_day"] is not None:
        n_unlocked = len(farm.get("unlocked_quadrants", ["NW"])) - 1
        if n_unlocked < plan["land_total"] and day >= plan["first_land_day"] + n_unlocked * 5:
            land_prices = [1000, 2000, 4000]
            if money >= land_prices[n_unlocked] + LAND_BUFFER:
                market.append(["BUY_LAND"])
                money -= land_prices[n_unlocked]

    # --- 肥料の仕入れ (ルートが肥料を使う場合。セットアップ期は資金を優先) ---
    # M5k: 購入停止 — 動物由来の肥料 (19個/日) でルートの施肥需要は賄える。
    # 購入すると $100 で仕入れて $60 で売る往復損失になる (実測 $45.4k/試合)
    if (not plan.get("no_fert_buy") and plan["uses_fertilizer"] and day >= 5
            and shed.get("FERTILIZER", 0) < 2 and money >= FERT_BUFFER):
        market.append(["BUY_PRODUCT", "FERTILIZER", 1])

    # --- 10件制限のトリム + 売却順序の最適化 (価格影響の大きい売りを先に) ---
    market = market[:10]
    # E018-M2 検証: 同シード A/B で -$3.6k (ノイズ範囲内) のため既定オフ
    return _rank_sell_orders(obs, market) if RANK_SELL_ORDERS else market



_ROUTE = json.loads(zlib.decompress(base64.b85decode(_ROUTE_B85)))
if M6_SWAP_WHEAT:
    _ROUTE = _rewrite_swap_wheat_strawberry(_ROUTE)
_PLAN = build_plan(_ROUTE)


def _m6_gap_count(farm):
    return sum(1 for row in farm["tiles"] for t in row
               if isinstance(t, dict) and t.get("kind") == "PLANT"
               and t.get("crop") == "STRAWBERRY" and t.get("planted_day", 0) >= 14)


def _m6_plant_override(pos, farm, plan, step):
    """PASS の代わりにその場に PLANT STRAWBERRY (移動なし=ドリフトなし)。"""
    x, y = pos
    tile = farm["tiles"][y][x]
    if tile is not None:
        return None
    if x in (4, 5) and y in (4, 5):
        return None
    if x >= 5 and y >= 5:
        return None
    if step < len(plan.get("plant_pos", [])):
        if (x, y) in plan["plant_pos"][step] or (x, y) in plan["build_pos"][step]:
            return None
    return ["PLANT", "STRAWBERRY"]


def agent(obs):
    try:
        farms = obs.get("farms", [])
        player = obs.get("player", 0)
        private = obs.get("private", {}) or {}
        if not farms or player >= len(farms):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        farm = farms[player]
        step = obs.get("step", obs.get("day", 0) * 24 + obs.get("hour", 0))
        day = obs.get("day", 0)
        hour = obs.get("hour", 0)
        if step >= len(_ROUTE):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        # M5b: ルートの steps[k].action は obs[k-1] に応答した行動のため、
        # obs[k] では route[k+1] を提出する (1ステップ遅れの修正)。
        rs = min(step + 1, len(_ROUTE) - 1) if _PLAN.get("offset") else step
        r = _ROUTE[rs] if rs < len(_ROUTE) else {"farmer": ["PASS"], "hands": []}
        farmer = _repair(r.get("farmer", ["PASS"]), farm["farmer"], farm["tiles"])
        actual_hands = farm.get("hands", [])
        hands = []
        m6_day = M6_GAP_STRAWBERRY_DAYS
        m6_active = (_PLAN.get("m6_gap") and m6_day[0] <= day < m6_day[1]
                     and _m6_gap_count(farm) < M6_GAP_STRAWBERRY_MAX)
        if m6_active and farmer[0] == "PASS":
            ov = _m6_plant_override(farm["farmer"], farm, _PLAN, step)
            if ov:
                farmer = ov
        for i, ha in enumerate(r.get("hands", [])):
            if i >= len(actual_hands):
                break
            act = _repair(ha, actual_hands[i], farm["tiles"])
            if m6_active and act[0] == "PASS":
                ov = _m6_plant_override(actual_hands[i], farm, _PLAN, step)
                if ov:
                    act = ov
            hands.append(act)
        while len(hands) < len(actual_hands):
            is_last = len(hands) == len(actual_hands) - 1
            hands.append(_reactive_hand_action(obs, farm, private, actual_hands[len(hands)], day, _PLAN, step, is_planter=is_last))
        market = _reactive_market(obs, farm, private, _PLAN, step, day, hour)
        return {"farmer": farmer, "hands": hands, "market": market}
    except Exception:
        return {"farmer": ["PASS"], "hands": [], "market": []}


adaptive_route_agent = agent
