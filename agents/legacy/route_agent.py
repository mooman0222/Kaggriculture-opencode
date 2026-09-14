"""Route-based agent (E016): #1 Crop Dusta の行動ルート + 適応オーバーレイ.

ルート: 720ステップの完全行動計画 (#1 のリプレイ 101701447 から抽出、base85+zlib 圧縮)。
適応オーバーレイ:
  1. 雑草修復: PLANT/BUILD が雑草タイルで実行される場合 DIG に差し替え
  2. ハンド調整: 実際のハンド数に合わせて PASS で埋める/切り詰める
  3. ルート外 (step>=720) は PASS

注意: E014 の知見どおり、素のルートは過学習する。このエージェントは
雑草修復とハンド調整のみの最小オーバーレイ版 (E016a)。市場オーダーはルートのまま。
"""
import base64
import json
import zlib

_ROUTE_B85 = (
    "c-rk<O>Z1mlKd-%&T~++n-n!STIv}ILyH1QZDZDG7zSn*3oK?2-n|XxzmFt}UDZ{Q84>wjiL%`%x4WsT_x&;>BO`zO=d=I%`ImqE"
    "{g-Ed{O-qR@2~Fep6&ne-LwDx`G5cRf1kek^gsXj`IrCs`~QFbzkmJVyQ6>b{fD>LuYdaZ&+b3mZnJ;+<Fhw!AO3=U`u@XTf4X}2"
    "_J^zMXCI&N>Sp)xKP&5(yEohId-=D!?e_JjUw+tL-@N<y;e7VX?#=e<{^N(2`&s|rzkT(W5ATngGYkR#ef;flbE5Cx-Q3>4dA6U9"
    "=%+TmzrA_=;nn@PnVV#=<{h}cdbNH0i%H{m+q?T`;|5GVP9|S$-YOf#;g7tgz!93g=TDy|bn@ic3XWd!&;;^^S69<*Fdt&@RDY6J"
    "{#x8#<zpyykWba<i649Z?ta)?N{44JJUlLV$L{KWdpqmP^v$eS$u0m3WatCz;X{iGYjJq)=+n@@@(R4W`7mE^voGr_@9-!xZfLZI"
    "Rq4)|vl2(6IPSMJQ;$P7+0*YvqcVN)VLd&Z<Kv$z%VKCedM{e`Uw&>}EBSlj=sC;gaq5FP8&<}A_Ij=jxI4^|gr^i5v{9#~TfFiL"
    "ysqOHHyui7RVQntZSjrHg)c^FhZQ%sZ^w@xkKp0Uhfnr;J8Q6ok?pzOYv&SdNVFluD+RASa97o>Fl>>$y1BmIzPkVEPutu3x7Tm~"
    "Q`jYFRXx6N{7GvnYQXWcAKy3rq&*v*e)xUf_vE$ku7(Dwv^LOW&PKV(cp1hO9c$*HgZ2!zwaTW!=)Cj}k**ITd)7VQW+NY*+HRL0"
    "2b^7-ZhOHwKfbBjpEiBO>0!Uihp`JHzWDF9*PoG@(ZBqJJz4&*Z&m;^X^wZ@-QQmAzTe*7{`Ir{P`DMxF3HUDIM;!XsTH)^+xWn6"
    "8M$@}@8<Btkr@zN)y?n@x*k$6tPx`wotMKHM!l16_xysJ*-u?YjogD6A`mZsba@Sru3JISL>o147%2h{S<4Ic2D-*m7q&9%864AQ"
    "evO`ow!VfR;HgD3FnS$)bTf=Le?bL2fR7#~au}(ZEmwK)=pm!EIx#*~lXG+%S%#w96E^1~%K$g6v;8r9byvIQxBA!)K#XdcA=fv;"
    "UkrmyKQEBC46n8b{b{~kEfXm>j2f*Kbk=u(GSKnW(>#HIM`UBE2HDrJF$vn$`cjDflYuE13)FtrEF5%mj27g~tZ8dBT9f%wZ?*%I"
    "aAc!Z3&W9S$dfb~!c*GtLmVl8a1VUyYe7|~$DSVhq>-*=jV@yL7)X$6Xj!qVr3|fxn~a!N-J*24sU<^pOIcbSpr|t+sTAx!ZMh<&"
    "f!N1&210I)G-DzpVAr=VqYE7agKw^G|0A@oAbc)cSjetIl#Vgx&=~6N4}L-u<}hdx4aFP2(c?IwCxmJeDv8<<0ijAw|AiOSVaSih"
    "A26Ka73DPp5gHIZXgU>)kSB+N;CesqJ)RFb1Iei_hp<Q81T#XmTd?SL`j^O?Q{(*lK-z%@=v7r|e~TKY<DU*Zt+QopR>i^pIC&r6"
    "BUwI~jaWbt(g^aV&7Ac>4uLnv>@Yka`pI;$H(64qy(=6_IgbjOx7xKY5aLV<8@~*-2^<hZY{g0ZSDD5u^hO&{a%l#)c+*H>vu_wb"
    "X{3tbx#?T_g=Fyte%M-tIY14a1%M0BB?NwID>qu9Alt&_h{K71xka|)sty=@+XAE2(ibvJv13wAK~M*;B^;64hj5+kyyeWo56F~%"
    ">c}v%16Qm5c3{<f<yp2oD^@fzNCvB%OX}g62?`hGQ<P|pC;>*L;I(q}a!?Oqw0gw-cx!bBbfLi+z!JD|s=-5iot(ZH8;{T;lg_i6"
    "r^B+-vcGIJ330CvD}Q%$bN$)JVV)BDSLd^RdpI^9{$QmtIB|4+xH=;*8sF{3D8o8DEid-nBKa&fUI~t*?)t^P!QsNp+Pq7Gt9#VI"
    "^Sw(1&GXP9`4JC!13@Xt6o<YAUloTVr6s;HdWZro;^Ptj!7Q1x&0@xb#nc7>{ya-QZ^{3jn(x4Yi%(V9IWbX}Pg~f=G4FTunAAI$"
    "TZY^ayqVD3h}$i-0EPj=!GH<p>Zk6|h2I2IH6h13np)BjIPBB{Z91T%e26qmwnShK9sV3%b5F{o(Lx!6SWss-_001tG?K+b`^{u;"
    "VvE%Lvc9C9jA#<JwmyvA(xz~36FS;^%Qb2FV&aI+n%S3LRO0qTXIm8l5jizUwpy}0Gn1@|*^KZ*D*B`m2^uzlu=DVcEEqY5sji92"
    "OlaNpNalkRm|J`dgKp-olYND!1`5$M04!t>u<i|TSbXuS+}vX05TZeuXHi%~l3mx$d#Vgd=GYeEr@0-Drk=RBw5fIDn4Vs6z~U^)"
    "rqz=V9K0>qLbBz>*dUQBi{y{?ikb-_M(EPuFU<^UlHro1&P1Qs05;`ABpEky%f4xU?<iz>vAEM$<Cht3o@eNL=6Anjx?1#SIIn^j"
    "nPP`FuuAdR61+cXswOJ?7$^GS?e$;2M3;;>C;I1jJh_YfxL>|JYIUaZoKdVZ@!gVEGg|VbffrfM_z?aMsG%@yYwjVo)p5wc4f<T{"
    "T>(jGrI8C6ld@dDB;p)g*uhPfdPQAggGQ>h4lQ*Wn}H^Wn^9$qt4t$eXa`~Px6yQ2ES|##BHGX@eS&CYTiHqKVqZsj9Mpd&Gnizk"
    "2`ML)><^qm-)AP(9Gvu^OX8S2MnP`Nn_<!3m>en0bPq{q2&}R6qJoTenNA9KDBG*{1|=|{b3#?Nr_29nI;;A{dFjiQ6glG6H(_s-"
    "(m|0yjfDLqo-p-^A&vKJ8CRJEyc7FGm5_4F;q;>SGF#j7VygkVb;T90=o|=h*-iH<kC;J5WOR$eZJbk=S}10z-aB%bnYF|Mq^*IZ"
    "T<}AfQRsZCQOi6}z8CW3l6N<f@rwUd9vqp`fG@?|ZsaJlJU}`LtgT%RSJ)jwn5W!JHaFIq+U#8sQEEluRv#TZ>0j4t(S=SGd9mtc"
    ")OASEp&pIM@N86*%CNb+2oi%fu#s)<*w#t$gE?wRWhTK_YgrFRoo~n?9F98lW<~Cz;*vGU3`3ZZArW->kkm#Cp2);qm=cC@{>rOI"
    "0X%vFR=PWGpdNjtnIg}FzMQZWvao$e38}50D?`qe4)tm3uT{P*>l3gcWI~za`y22VI+QBF{-I7^KpN#{_091v_$HB)vln|C`0;TR"
    "UE;A#snIXkcHWZ7SP1b&CPt^lZ?LW996-5*++xIC2KiJ(KD!EK647b||D;3=gR?=cs4kfVh|kaNNQO4nsq&T&Pv(Xf&B3N)EZ3Kk"
    "T)GlR{4{KW;!zl*B468UA_E!}T#!cZ=<cekT6oxpz<wLL_4@Yay+|%Dl~Eck#6z-tx0uD;yVtpPq}CB9z4g=zW+k8;D4DI^;ma6("
    "X(L1%N!-KoGdh!l%84ig#w%!?Cs*hld-s9Ynx{Lc(-5D3FZb>g3A&REHd3SLLKxt&2`~I2S;XdqI{+>^xM*L${j+nR^_iR8h$BZ7"
    "+dB%bBzA)xTI{+fJrNHSwy4%Gl6zx3g{)K>OMNBf(o5N99TpqI5RPKkSA&Z?Ka+BTH9Fabjg}V??9ffdt<*xuLZ1W(f=uOMWnOB*"
    "b0WjwPm585(V7F2xIw1CRDF<cn2!5oX{>Gf=}FME9Ayh@`@+q`?J8o&Jvgpw51o90Wx8M59xl9-;BAoaN=X5yVcqUjR~F((<n)j!"
    "I3|wwvJuEDK}LjHl+38ut&0ZGNZ2@t^imQ|Ah}S7P4m1^)F!R!15xEro=Q^^=)83}jE6RlnQ)kQtiG)_`+5<d4qBN`{))N=*KJvB"
    ")H6k!ilUcY1cZD83f)7KJwp|SjG&Xb!CsW|VVNW%(SHyLPfD8aZ0TKyHE)cJU53rk9n@u}UN#`9Y_~fyk(5hg1p!2FioTy13>5Z~"
    "nJ^kw0lCRUmV-6iVXY{2lbQgdyn=NEq&9(Mgooi75qat<&#`HoAfDT(;0`$fhWIs7fj((6+D$rLdlTb}M(WsB;g|K^E+KhUq`9`v"
    "kal%m`n{n^`-nw_YdtT0juHG`7P~=nk&wmOvR(=RKC+EbKS~o49q~v^Ra;>v1h0(jc?e$aDykWac#$T+UXz}JPK=5Ob|TTLHv~p}"
    "nSh&0>YxT(BVO%+x$WZzD0os{DG!@Lu##S0i-_f>Hcwq%fe*EqWPhPvO!2XHD$1%<7l$Kbqfh1;LPSj^W;(q*bW+%81-d<n4-Z`i"
    "x)0$<K5L*kMx{X{EO%Q9KF$~-O>rE9)<zGEBwoB23`4TNaV~B+5saxpQp?B{u)+EYsudbphr<vYXNOhmkdchIwA_cymR372m|#kM"
    "3zkhI<PLhGu9e-(27s%Pi1|%o<rcd}A_28NGsvV!0uDVJc&l0wb9P7sLr-!Q6-NQ#_!fIw(hu19X$ZkCfOEhjW4iiN7Zng<^N;Hm"
    "=bUz1t6eOO{m9}{t8tS;iz2<W&R>o1ZK1w66Qkl0$e<__oE<?Iua9nhl)P_9Zsr2#`2pONut}u!=^~N^@zduwPQa9g>@9ypzsO=m"
    "h1x8Vew8xF@RVw(EO!C|y90z=-%4gOAL;xay9r0RYK@e?&6$ffX$sN`Yc4oNsiz33kKIrx)M_IgIX~E8x%Os%*dhg@{z0gPb(-A-"
    "9{O_LkR)rd0#Qz8Qr>RcOsU(OCA^q;2S*KS9H3@#wVam4g7sb*DP)Le3zmYNXKmt$#4#|=3V7pp4QN)l(0Wh>p7q^p&8fAdIpL8q"
    "$l}An7yuP>Z#f2WIb>84yp}TC1IY_Z^cqldUJ4qw9oa9{B)_#yZwg==Pcwt#GH_rx4smS5Dkmd>TXMA|OAj^}Lx|#V0Lomwk&2Bj"
    "1Q9Xy!)PVJXk=kDn~VevWrvt-@!om(L_??+nM;FFmS_IR#Ug_7K!c+dEoq7>MR+s5?kr7oyHbID=fs6(&iwA979p0xWE0nf5R=tH"
    "H}9%REF-Ve#NxpcueF{VrZXaibb^M7icJkRI~O*`n<YRHf`?U#taW|}%~J><Zq@8%WW?DQ-NlI#`8BAL)UPM5nKI^v@M5pjW?9Bh"
    "RUKS1Ex4i`N0~3m;n?ILs@NDz^UFDr)oOjsP$t60VC)H@e25+((dlY@PKA&vupe_4Rzp2SK^i~H#$AwkoC6bVHn5gq;zR11at{wY"
    "m)!zZGQJ(FWy6vmpAB;uFj#ln0)Cs}=sIZ1imj&MEniGQ@a`1cqJFCOMnVzDS4`{H={<e0(D_xz8PMq3FeI9ih1!zB4(>Or*b_QB"
    "Wf_^ag?3T7-k3~OEe>f^4C}D~k?R%)MFU5UbYt?6_+$Z(uRxZ0m&m1Lp$g<&AY~(#%d5;GNkdvet9{acx)`L<+L<)u)p=g>tb3y>"
    "m!D_dYdXj>s4OyvmC!yf1`tX|nd08CzcaVAGB;u5lt{1{rO4=@OmFE0B?t~=PJ&{<{A%MgXqb@`Y}g$(Y)Gpn-qaDFas>&M>$H@K"
    "8&-5uv~%8YA?jkKNm5R#;e6(f06kQVGAcNaFQZhIa`Db=Ba@uRFv~gg76NjBs>a?Mh(*3OydTh~DAunz8eWGsHa`&omhKs332~OH"
    "P-@zL5GdiRCzz^I$+pzk_$ulTe2)aAFre~4^n{`#sCT}0ylL>rkJmI$pVHDr$tk*e-bIsqCma;&@SC-PjrkkD*AU!ewlC+=ARHs+"
    "HV{&oN|Amcqv?bT$iUZHP3LjF6jYF+)s<(YSqVU=t26$mRl4e>!+m3s9}#p_)~i^gwBxcAsNOlYF4hChF3irneST3RWSv$JC6R??"
    "`a)d-Q$JdjV?hDsUh0r6KF!dX>7k~WkR6NHl%|QWyeuO*g>V;P|5~JHcy?$-IIJnv5O!Hm5{Gef?k7CM9ffl@&9RlS4HF&t+AC%5"
    "r@+Qn(E?`t^Bg9X{`E<1@|(a&=ihJ29?TIcdf5^GJdg3u^X`(4OvcP%WiFG9EppQonTNpB7qh+D9zNUn6}nR-S_z-y=^4tR<>_<X"
    "i=@ewzIFZ=Idn5SsvCI3mk3cq$)k&cosM?jVpL(crG&EZINq2NRoW&i=LR1W!W=nJPDToLSZ4Ng-)s((811YO;-r}`peE%J^A@N{"
    "btBj!fJx4g_gP6+CVnBe=0H`I_GklFHrvDKS);n$+hR3jQduNmy;byLR8C5Z05jcf(MbT)1*25gBUt{tmDn%E9oqIDV&Na1XqG`E"
    "<o+<2RIYPt2Nz@rEL_vNuwuu&AT(=2vP&4aOGF^&VWu?aUY<0Y4h1P>sr}*9!OE7v!?~$n_JRN;%j0Dhb_uc90#Q`ek2CKWS(Y>|"
    "QSPW`iT5kdWZY*ICd6hL2X{O0R|%CcXc~)HWpT?TlyU^gQJSt4DWxifY`YzAJl$JJeT$>sY>G5V&l7<8q$-)Y?RNxUo;^P{0mtNw"
    "2-KWwgqZr7t974hE$tr@MTNyGwivEj9*reeMZ4x*wQe>ksN+mfH;DG!g~D+CY{1YYI1;bzB;_HmdI+}W1P7bADc73GaqDGNwm2b`"
    "1>z+{))mLmtY;!k5fm2C3$PefT(ujY@DZ!HGSezd)%_~KTVOsqVew%;a;FCNW5C{=%AXw&4!i?Mk&Qf4k7dM@B_%>mM^Ki4-wT%u"
    "ua=wFGSGA*Z8(FMG>O_ZLUb8iiq>onY$T%V5N13ELinZ(c!>&gHySaWqhbpw3IclbA+`z>2}qk>ktI9*S)B}Wq6Fw0!vl0V334n-"
    "A!>Bik-Tu4`$^>}%_Sttl~{x=Qn`a?z2#}m<@sRjfGq9Fxuiu@s$%sJ@_ylGXR?y*Q>ep2;nabQ$?<Xhs0~awH#;qOyRvi@VUc{Q"
    "6rfo#3i3=5CgCk;(K2Ex@l;RX-&3#`S3PAS6Jnzb3_9g=o_DIZO%Z!ZB6<lyMqN&fgymSjaXO#MDJ&&-pZjFTydujI(_An;2tv!<"
    "eG*nx1mYyS$%183^8&pzTFyjENGlAz)1cy+XP5p#c0^@j8k!!=m{R4GOqEpY4jF{}aD|f=ul=Y8sjj`;g`6hJj35Dfan!P=jL9`^"
    "y(JhM!6C+NL3@*gocuz%M0QRWEn!A*L0Nstnw9WkJci(anu6w+aG?QHeTGNqz_oes3dfn42MiJp`)K`&Kw~K;B&3-0kfU0MgN<Uf"
    "R7s(5MRb91AeqGoa=%&wf=K>6%UUAbXe0ezYOa+iuUv=B{vg>en%$L$Q-X65pQmVhYk4_FadLt#hmfFgboHsrDeTT*#W0xOVZFrC"
    "Hc_y6(eoz4jPsZsrQXVPWKJ%4Y=YJ`5|!W<MgZ4#tGRwc-MXW0El#Jwz*`}pZ4E=F=N|lq(>pHe#j8u0jB(0pOFgI85tCcjWeZc|"
    "%0jzmw&)z11jIaZ#IL&42K43gEG!4rLD`CdY+}mS=ZXZ#tfXz;b0Z4PDRpfHscDoIb0wYe=vx=Pq);P4wF=;}BMy}zC#UyMS6@>E"
    "5JI+zz<40FQVMPgXCn|nbtWG{@VWuQvN0}!QJ9GrDUxsT45^iZORiY1DMAfjVId{we9>zmsZN2gt*R}^%Qi8A&dNj_Rw>}&IZ1)k"
    "LJ$}phS|IEE30I*T1H*a@GqHLCa<`h>!&kqgDY4U3N;f4TPUkm4WrP?W@L6k#L?0Ut3}GG?4ypPMq@-Ws7L(+!mU~_a)}_7Kq2fV"
    "i9y(N35K^LA+MAD#CXFtC`$|LIrT&pX<P}qZ9<Z%>K1es5RY(>-*cp%#AWm4g)GT~NraD2p+j@gCkoY8FPIbV_gi4wt=2ILsOI6W"
    "<YZgDh%?LV<9rYT(R9g&)3cM9)?uY=XJ^bzhg%85PAy~KN}NL&T@G9tJQEVbmdD-@#?==aya&omyckh}u`aq28?^r@1%fj6D1PCl"
    "#1gK*ZjrmkI5P%Ze!RfyQrxCAuRfTlIRh=z2$?O?CcBVi(NjOEz&z{=^^==L$`&{B$>PhaXenYYWaJ&;kw*tn{EReNE2)aryn)0j"
    "87a9yq}4;OfH6I*>`0nI5G*#Mlvd`oH&S+#SeY6+3DTRB6f+}zxrGB0<0Jz7EtgHviVZA-s1USG0*b0uUsJzGtzlNtHc$79Pr*DS"
    "P~TbP72=*&l40?YUa<z$gm0pO#CV5<<)%mJ^avvvVyZQh772E7ca@z>J3wQkX3BRS;xQ8bIgelC&z@LyzoKfz7NEB2kiU8%>m%o`"
    "wZV(q-j%|5?!wO1!ASiA=4;c8ql#F*R5x;82CPV8s$jcv{2;|uJEMq%NaiYw$NwW<hf4r;Xwz}@jZ%QfP3cKOiI|Wo*5}CUu%cYZ"
    "vPv0*g0_+1n{PEz(#)eKswv=@acbSl648DPw@AH#8%wO$A(i00lmkOT@-DM_C0V%SUa4B}0=dkqkg4C*yH2@%Oc#!(N=q<=rc_`F"
    "d5UKSW!T6^v%w5g<@}cfkeFUT4Xfk9J|ckBDT-l~CM6WoowIn=Xa&#7lg>RD4W#(9tO)0drbWjcz6zx}SweA}jJGV?^V1Ep(TsU`"
    "V>XGbDUzExd$}-&MwS#b0u2l=)!AU8xtIORpCc5EQ~t97t8IN&x97l7vk0!b^q+~P$ugptt(I<qTE%)TF|KOy2*8o1>@RQx(5yeK"
    "i4$P>t<&*#YFYYYybu<`cwrInHm(xh`MOKtSi(%8Jg^keXk_MXCGg8*?nPH|hxEr{5m;7d3uo$XJx{Vtf}g906Iu=>ADFOx$i`_z"
    "00Ur*T)r-<8~X`?6N$#ZQ)NxUx(fy**9B1#C{2dVaTeiQNq#>%aNu)>FOjId=5UCT!W@aV>SL<8eAVnbU5%T_c`Av~!?rMJbA{c4"
    "&xfggKB`@6MNKF)zzg`vrJ~K)K~yC!vnM20pdK7+J{)xFLa8L=P83F~TNO+#smEfYoU6pXA~TfcN};6`-!qi>MG&ju;P3?lLq0&4"
    "TwZut!F@<FJGo|lTye69z_SPg+)OBCPZrtFz4JEofJtG5-1wxydN!_*RNE8{0-I5$jXH__azz9c^~|>q<7xBSB($W#R@Sa<fvGq{"
    "N3TCUvQ|OfuB%%KZy~1dO@%z<>M)^BW7}?Ly`s_RdA!Le(tIk7`}@q-z3|d?XB5J-jC5$ap+t&DulBC7t~FIdzr+WVm)0xjsuHSw"
    "^TXtnJfrYC{?o%{Q1P}>{D{CTt_rPJE7+>Lyml=2ddg*$8-$EwOPOmDY;21tJE~`LzJ+K#S6z~nw>wq8k?QhQ!?yjjx*#dnSCgki"
    "?ZD8Y+>9Qi+`KfQF-bxVZAOpip9ZHWS7~${886;BBp#PUUuCJq+$gw7Y3HMX3J1A@eE13+M8B5wgfahhB~Z8Btdy8s`WGFvLRxo~"
    "txM{Ph%T6w_|{2U5})@a*>H~IU{Ie52}~@=3S2<L38K!lkWy*U)Wo{EtWQ(>74y-IJvG>Y@%>eq2QRH&<~te{Q`0rr6UzxKPGhwb"
    "nQ221-EFa^Oev9c@;M|af^EH^;Y~S9-q6i#;Bd$BOImnSZeON`c7gzZqY$8+MSN)W%NXZd?;9vwJ2S#-<ODHNLS4m<(@FHi;#!H@"
    "1|>uuLFAD?CZQ;0QhhW^Zf^|d8x_+a{PZuc5E!LOk%un64KJ8M<(JjOq1d|Vq`4=h-4GCY7xDN)@)b7|)krRp;zLd|3`ZDd1p>c%"
    "sfCT(79eK`ArBLRwsO5QUVmqreuKhJ=W@Zhpcvs}_hm-CUurc2u_}FbWgrkIgzr2eqQEE?q7@G@uI6I}or{SWI!}|P(Cw~zcF1Wt"
    "RNNqT<ufHox&V7+zt(CN>nWtfe2x1PoEL#!==QF_C*p9v!!DY2YMwTb*-02v*^0GeEe2AEU>~5Nnb}pg{z$|^uc<jU01b+jQ3}DC"
    "!1caN%-6YAJX~rN0OUKak^>BL<4eMGOZ7kqJ)&e4r@$ALtSBpm<g7g*X3Vw^KTIeoj!z3Y+E^PIs?UY@GQI4648@?J9avn>sCO9u"
    "6T0^+NB3Un=-xU;_cjfp;;h;O$stXA<!*BOgiDL1P<X15t^nn*^>Qu|E?!DtC94A#g49v@57(@qGKI{(mxxGxxs_dNHdxLGBJg4F"
    "Qi3m36P|3lQv52%E-=<x$mN#0*SiXD8LT+WuUb{63ctsQY-IJqsWD<d%*0bsf^gjyG~7^x_-N(jlZ4kj+M_hLM{sO%acyoEixe-@"
    "<94>*+SblOma{Qa`kj%L)@YY$_u^4b5LA)GcXt=1WoBEBmD^xlfU|2%Ia?<xC)Gn33<i%U_ecc}1j`L83t0vLo7KQ^rT#2K`HbVq"
    "i&CHM%cQBGeWl+8Lm|x|vi9>N6)K$n&Ax^0C-BVKsDOl(L^efS?u9}*xhOR?h!8|-09{7ea&I%Es3Yr)x*)^)A>7rgcHLgAQSsrp"
    "=y#rioav0(f^XV%;bn&J&O@<ZB0@AQ(lnDVBX<-RtINF|o9#pIa!3cx%t*=o@SM>GpA<~)-@6kk`@7*>UL-P%XWcoM5DcSQJS2kL"
    "P3f=3xZDYJvfSlbf_2Hs@`iMJK`@+vtx-J=Nef0oDytZ)XR}!Ft6HA}MXr;y8|5<8()s&<k`iGPTA=|AE5&O|_?=tr(BblKvjC4S"
    "09xF4q#Pv)hCz8#0VLAsS%o>6taT-kwoE-wlI^OSg^6ngjws6JtTHMX)+pqlxjocFD0~yu3#|YUH7GD#s*fW~qb?SsRw5?2$~wkX"
    "N`hQg9;d7!Z^S`6pbCYYLx$i6!=jg$c&g4yV7?A5N_C7SJ4_16<bfu34iZF9uZ7AWsdn7KMuRc)*oaOL)hmW1*`(@q1^{coAOcwM"
    "#RGsQY+jPCTWE_oaT6#f_Pbzz@$V$f2J|9UpB$DBYc@IuhhM|!J?eJyWySg>gN(p79uyh=cB+tl_+g-|OHJS2Y2z#i7@$8Uqkw+K"
    "3tqj8P9Wm16+}GBF-Y#L3FvqsgN~ca7<4>upn_5&n~|wv2;l0`O+U4Pv0IHwW*L{H(LP<E?6go2!pPCJYwBcKS)D1UeqBM9`Vt2f"
    "8QF>^kOjedah_!p-YZ{bDK@5L)~a>Ki{x?gZR9q=6|YF^$<-w*XH=$Z##{$PpYynAzMwjz3NZnn`c$eXOjRmamoufjLL;M=(Z!rj"
    "1(%3DhqPA;t|&}&jCSb6;nDyE$BaUAGxCu0B)du;JG}Qu-YJb*^4bBU3SO^9XFMq-Qi6wgn}r~3N_iGJ39b((Op<`Vm2+8Q0;t(s"
    "4CEusJvZbLj;F%oGQ>l6wv6c9i^@?l^j);+uzE`c)KL^$EJX=|cObZm=yAn^)l?xbp96$bk#KGuNDCui`nU?{6j|)$VP;v~={U0<"
    "hTFJ4+B%z1|8l7%e1i45!+Jc(-^K=9eN{FfwT9&QcyQb2c1guf!cFdg7p1m}rU2#SfO-OUI}dbns88;!<n$}&eHNU{Md`#cRgcEc"
    "!;u(IknlnRqHuPT>zg@ZZ<>yxmqiZ!35tof>()@P!z(G8xNg0?qR5@4WLnEp9*Tx4rCfBFs=NF&m4XZPl_>CQyVnd?my?SyUrosN"
    "5eZ^0VN?K9a&jrz7eqryT}5JX8zTGvFzTa(!%XfeTDJL=6cngft7u-U1OsK&LK<r>Fr<v%01A<Jc5d~=(|59hVgAWioof^UD~&zT"
    "M8ca*sgg2Z&6m<fMp5j^hLFcS)iP<ans_a0G_c_<C7+?n$fNcex`##;Mq<lKAVc2T_z(;K2IWy2SgX`%5-$koW4IhO`Wda^;m`w*"
    "uU(~aqNzQ?L6>!|+2fHt@XR(o;Pd2&6!+mt1c(Z+YW`yjZwbUFkG~vIw`E1Z=P}x(l4nQ=$|H|7Lk{iyuw+PQ!*5E!(BN4$(~FS}"
    "R7snqhJ`E~o&mgd^K`_e_YE!=y*d~PFp%`&U%?8Icn1va)hnuejdkI2hjy~ylfcUe=Kf&I)EPrr58>1jsXRWzF@1;$iPdRASueSX"
    "!Iqy<oDoP44tA(fjDT%Z$2cG`3a8J8N>Dh5g=7`&nmXPElHjOHZjldjd-S1OpU8|coQ0)Hd7rxV>dRmSusug12_Cax|K@(xmuJxV"
    "31GM<ogsE!pw6+#2y$_sW@->VG7LOL`V75Jk|LH%-DT2!c~U33PtwaK|J2lU<hs7F7F*s|IUlcat_TWX>n!H*5WG?j69?z_@Lj`n"
    "y<BYc$&BQGsE+;1hOy7bujOAn{GFsZHb?9(t*!FdB-YubKZTt=VswQ>sr|nJ{1gpt"
)
_ROUTE = json.loads(zlib.decompress(base64.b85decode(_ROUTE_B85)))


def _is_weed(tile):
    return isinstance(tile, dict) and tile.get("kind") == "WEED"


def _repair(action, pos, tiles):
    """PLANT/BUILD が雑草タイルで実行される場合 DIG に差し替える。"""
    if not action:
        return ["PASS"]
    op = action[0]
    fx, fy = pos
    tile = tiles[fy][fx]
    if op in ("PLANT", "BUILD_COOP", "BUILD_PASTURE") and _is_weed(tile):
        return ["DIG"]
    return action


def agent(obs):
    try:
        farms = obs.get("farms", [])
        player = obs.get("player", 0)
        if not farms or player >= len(farms):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        farm = farms[player]
        step = obs.get("step", obs.get("day", 0) * 24 + obs.get("hour", 0))
        if step >= len(_ROUTE):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        r = _ROUTE[step]
        farmer = _repair(r.get("farmer", ["PASS"]), farm["farmer"], farm["tiles"])
        actual_hands = farm.get("hands", [])
        hands = []
        for i, ha in enumerate(r.get("hands", [])):
            if i >= len(actual_hands):
                break
            hands.append(_repair(ha, actual_hands[i], farm["tiles"]))
        while len(hands) < len(actual_hands):
            hands.append(["PASS"])
        return {"farmer": farmer, "hands": hands, "market": r.get("market", [])}
    except Exception:
        return {"farmer": ["PASS"], "hands": [], "market": []}