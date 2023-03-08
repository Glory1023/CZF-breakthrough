#include "czf/env/czf_env/breakthrough/breakthrough.h"

#include <numeric>
#include <sstream>
#include <iostream>
namespace czf::env::czf_env::breakthrough{
    /*initial board
    ┌───┬───┬───┬───┬───┬───┬───┬───┐ ┌───┬───┬───┬───┬───┬───┬───┬───┐
    | w | w | w | w | w | w | w | w | | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
    ├───┼───┼───┼───┼───┼───┼───┼───┤ ├───┼───┼───┼───┼───┼───┼───┼───┤
    | w | w | w | w | w | w | w | w | | 8 | 9 |10 |11 |12 |13 |14 |15 |
    ├───┼───┼───┼───┼───┼───┼───┼───┤ ├───┼───┼───┼───┼───┼───┼───┼───┤
    |   |   |   |   |   |   |   |   | |16 |17 |18 |19 |20 |21 |22 |23 |
    ├───┼───┼───┼───┼───┼───┼───┼───┤ ├───┼───┼───┼───┼───┼───┼───┼───┤
    |   |   |   |   |   |   |   |   | |24 |25 |26 |27 |28 |29 |30 |31 |
    ├───┼───┼───┼───┼───┼───┼───┼───┤ ├───┼───┼───┼───┼───┼───┼───┼───┤
    |   |   |   |   |   |   |   |   | |32 |33 |34 |35 |36 |37 |38 |39 |
    ├───┼───┼───┼───┼───┼───┼───┼───┤ ├───┼───┼───┼───┼───┼───┼───┼───┤
    |   |   |   |   |   |   |   |   | |40 |41 |42 |43 |44 |45 |46 |47 |
    ├───┼───┼───┼───┼───┼───┼───┼───┤ ├───┼───┼───┼───┼───┼───┼───┼───┤
    | b | b | b | b | b | b | b | b | |48 |49 |50 |51 |52 |53 |54 |55 |
    ├───┼───┼───┼───┼───┼───┼───┼───┤ ├───┼───┼───┼───┼───┼───┼───┼───┤
    | b | b | b | b | b | b | b | b | |56 |57 |58 |59 |60 |61 |62 |63 |
    └───┴───┴───┴───┴───┴───┴───┴───┘ └───┴───┴───┴───┴───┴───┴───┴───┘
    */
    BreakThroughState::BreakThroughState(GamePtr game_ptr)
    : State(std::move(game_ptr)), turn_(0), winner_(-1) {
    board_ = { 0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,
              2,2,2,2,2,2,2,2,
              2,2,2,2,2,2,2,2,
              2,2,2,2,2,2,2,2,
              2,2,2,2,2,2,2,2,
              1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1,1};
              set_all_possible_move();
              set_btpm();
              chess_num[0] = 16;
              chess_num[1] = 16;
    // board_ = { 2,2,2,2,2,2,2,2,
    //            2,2,2,2,2,2,2,2,
    //            2,2,2,2,2,2,2,2,
    //            2,2,2,2,2,2,2,2,
    //            2,2,2,2,2,2,2,2,
    //            2,2,2,2,1,2,2,2,
    //            2,2,2,2,0,2,2,2,
    //            2,2,2,2,2,2,2,2,};
    }
    /*
    
    */
    void BreakThroughState::set_all_possible_move() {
        //black
                                 possible_move[0]  = 5648;possible_move[1]  = 5649;
        possible_move[2]  = 5748;possible_move[3]  = 5749;possible_move[4]  = 5750;
        possible_move[5]  = 5849;possible_move[6]  = 5850;possible_move[7]  = 5851;
        possible_move[8]  = 5950;possible_move[9]  = 5951;possible_move[10] = 5952;
        possible_move[11] = 6051;possible_move[12] = 6052;possible_move[13] = 6053;
        possible_move[14] = 6152;possible_move[15] = 6153;possible_move[16] = 6154;
        possible_move[17] = 6253;possible_move[18] = 6254;possible_move[19] = 6255;
        possible_move[20] = 6354;possible_move[21] = 6355;
                                 possible_move[22] = 4840;possible_move[23] = 4841;
        possible_move[24] = 4940;possible_move[25] = 4941;possible_move[26] = 4942;
        possible_move[27] = 5041;possible_move[28] = 5042;possible_move[29] = 5043;
        possible_move[30] = 5142;possible_move[31] = 5143;possible_move[32] = 5144;
        possible_move[33] = 5243;possible_move[34] = 5244;possible_move[35] = 5245;
        possible_move[36] = 5344;possible_move[37] = 5345;possible_move[38] = 5346;
        possible_move[39] = 5445;possible_move[40] = 5446;possible_move[41] = 5447;
        possible_move[42] = 5546;possible_move[43] = 5547;
                                 possible_move[44] = 4032;possible_move[45] = 4033;
        possible_move[46] = 4132;possible_move[47] = 4133;possible_move[48] = 4134;
        possible_move[49] = 4233;possible_move[50] = 4234;possible_move[51] = 4235;
        possible_move[52] = 4334;possible_move[53] = 4335;possible_move[54] = 4336;
        possible_move[55] = 4435;possible_move[56] = 4436;possible_move[57] = 4437;
        possible_move[58] = 4536;possible_move[59] = 4537;possible_move[60] = 4538;
        possible_move[61] = 4637;possible_move[62] = 4638;possible_move[63] = 4639;
        possible_move[64] = 4738;possible_move[65] = 4739;
                                 possible_move[66] = 3224;possible_move[67] = 3225;
        possible_move[68] = 3324;possible_move[69] = 3325;possible_move[70] = 3326;
        possible_move[71] = 3425;possible_move[72] = 3426;possible_move[73] = 3427;
        possible_move[74] = 3526;possible_move[75] = 3527;possible_move[76] = 3528;
        possible_move[77] = 3627;possible_move[78] = 3628;possible_move[79] = 3629;
        possible_move[80] = 3728;possible_move[81] = 3729;possible_move[82] = 3730;
        possible_move[83] = 3829;possible_move[84] = 3830;possible_move[85] = 3831;
        possible_move[86] = 3930;possible_move[87] = 3931;
                                  possible_move[88]  = 2416;possible_move[89]  = 2417;
        possible_move[90]  = 2516;possible_move[91]  = 2517;possible_move[92]  = 2518;
        possible_move[93]  = 2617;possible_move[94]  = 2618;possible_move[95]  = 2619;
        possible_move[96]  = 2718;possible_move[97]  = 2719;possible_move[98]  = 2720;
        possible_move[99]  = 2819;possible_move[100] = 2820;possible_move[101] = 2821;
        possible_move[102] = 2920;possible_move[103] = 2921;possible_move[104] = 2922;
        possible_move[105] = 3021;possible_move[106] = 3022;possible_move[107] = 3023;
        possible_move[108] = 3122;possible_move[109] = 3123;
                                  possible_move[110] = 1608;possible_move[111] = 1609;
        possible_move[112] = 1708;possible_move[113] = 1709;possible_move[114] = 1710;
        possible_move[115] = 1809;possible_move[116] = 1810;possible_move[117] = 1811;
        possible_move[118] = 1910;possible_move[119] = 1911;possible_move[120] = 1912;
        possible_move[121] = 2011;possible_move[122] = 2012;possible_move[123] = 2013;
        possible_move[124] = 2112;possible_move[125] = 2113;possible_move[126] = 2114;
        possible_move[127] = 2213;possible_move[128] = 2214;possible_move[129] = 2215;
        possible_move[130] = 2314;possible_move[131] = 2315;
                                  possible_move[132] =  800;possible_move[133] =  801;
        possible_move[134] =  900;possible_move[135] =  901;possible_move[136] =  902;
        possible_move[137] = 1001;possible_move[138] = 1002;possible_move[139] = 1003;
        possible_move[140] = 1102;possible_move[141] = 1103;possible_move[142] = 1104;
        possible_move[143] = 1203;possible_move[144] = 1204;possible_move[145] = 1205;
        possible_move[146] = 1304;possible_move[147] = 1305;possible_move[148] = 1306;
        possible_move[149] = 1405;possible_move[150] = 1406;possible_move[151] = 1407;
        possible_move[152] = 1506;possible_move[153] = 1507;
        //white
                                  possible_move[154] =   8; possible_move[155] =   9;
        possible_move[156] = 108; possible_move[157] = 109; possible_move[158] = 110;
        possible_move[159] = 209; possible_move[160] = 210; possible_move[161] = 211;
        possible_move[162] = 310; possible_move[163] = 311; possible_move[164] = 312;
        possible_move[165] = 411; possible_move[166] = 412; possible_move[167] = 413;
        possible_move[168] = 512; possible_move[169] = 513; possible_move[170] = 514;
        possible_move[171] = 613; possible_move[172] = 614; possible_move[173] = 615;
        possible_move[174] = 714; possible_move[175] = 715;
                                  possible_move[176] =  816;possible_move[177] =  817;
        possible_move[178] =  916;possible_move[179] =  917;possible_move[180] =  918;
        possible_move[181] = 1017;possible_move[182] = 1018;possible_move[183] = 1019;
        possible_move[184] = 1118;possible_move[185] = 1119;possible_move[186] = 1120;
        possible_move[187] = 1219;possible_move[188] = 1220;possible_move[189] = 1221;
        possible_move[190] = 1320;possible_move[191] = 1321;possible_move[192] = 1322;
        possible_move[193] = 1421;possible_move[194] = 1422;possible_move[195] = 1423;
        possible_move[196] = 1522;possible_move[197] = 1523;
                                  possible_move[198] = 1624;possible_move[199] = 1625;
        possible_move[200] = 1724;possible_move[201] = 1725;possible_move[202] = 1726;
        possible_move[203] = 1825;possible_move[204] = 1826;possible_move[205] = 1827;
        possible_move[206] = 1926;possible_move[207] = 1927;possible_move[208] = 1928;
        possible_move[209] = 2027;possible_move[210] = 2028;possible_move[211] = 2029;
        possible_move[212] = 2128;possible_move[213] = 2129;possible_move[214] = 2130;
        possible_move[215] = 2229;possible_move[216] = 2230;possible_move[217] = 2231;
        possible_move[218] = 2330;possible_move[219] = 2331;
                                  possible_move[220] = 2432;possible_move[221] = 2433;
        possible_move[222] = 2532;possible_move[223] = 2533;possible_move[224] = 2534;
        possible_move[225] = 2633;possible_move[226] = 2634;possible_move[227] = 2635;
        possible_move[228] = 2734;possible_move[229] = 2735;possible_move[230] = 2736;
        possible_move[231] = 2835;possible_move[232] = 2836;possible_move[233] = 2837;
        possible_move[234] = 2936;possible_move[235] = 2937;possible_move[236] = 2938;
        possible_move[237] = 3037;possible_move[238] = 3038;possible_move[239] = 3039;
        possible_move[240] = 3138;possible_move[241] = 3139;
                                  possible_move[242] = 3240;possible_move[243] = 3241;
        possible_move[244] = 3340;possible_move[245] = 3341;possible_move[246] = 3342;
        possible_move[247] = 3441;possible_move[248] = 3442;possible_move[249] = 3443;
        possible_move[250] = 3542;possible_move[251] = 3543;possible_move[252] = 3544;
        possible_move[253] = 3643;possible_move[254] = 3644;possible_move[255] = 3645;
        possible_move[256] = 3744;possible_move[257] = 3745;possible_move[258] = 3746;
        possible_move[259] = 3845;possible_move[260] = 3846;possible_move[261] = 3847;
        possible_move[262] = 3946;possible_move[263] = 3947;
                                  possible_move[264] = 4048;possible_move[265] = 4049;
        possible_move[266] = 4148;possible_move[267] = 4149;possible_move[268] = 4150;
        possible_move[269] = 4249;possible_move[270] = 4250;possible_move[271] = 4251;
        possible_move[272] = 4350;possible_move[273] = 4351;possible_move[274] = 4352;
        possible_move[275] = 4451;possible_move[276] = 4452;possible_move[277] = 4453;
        possible_move[278] = 4552;possible_move[279] = 4553;possible_move[280] = 4554;
        possible_move[281] = 4653;possible_move[282] = 4654;possible_move[283] = 4655;
        possible_move[284] = 4754;possible_move[285] = 4755;
                                  possible_move[286] = 4856;possible_move[287] = 4857;
        possible_move[288] = 4956;possible_move[289] = 4957;possible_move[290] = 4958;
        possible_move[291] = 5057;possible_move[292] = 5058;possible_move[293] = 5059;
        possible_move[294] = 5158;possible_move[295] = 5159;possible_move[296] = 5160;
        possible_move[297] = 5259;possible_move[298] = 5260;possible_move[299] = 5261;
        possible_move[300] = 5360;possible_move[301] = 5361;possible_move[302] = 5362;
        possible_move[303] = 5461;possible_move[304] = 5462;possible_move[305] = 5463;
        possible_move[306] = 5562;possible_move[307] = 5563;
    }

    void BreakThroughState::set_btpm() {
        //black
        //                        left                         middle                          right
        board_possible_move[104][0] = -1;board_possible_move[104][1] =  0;board_possible_move[104][2] =  1;
        board_possible_move[105][0] =  2;board_possible_move[105][1] =  3;board_possible_move[105][2] =  4;
        board_possible_move[106][0] =  5;board_possible_move[106][1] =  6;board_possible_move[106][2] =  7;
        board_possible_move[107][0] =  8;board_possible_move[107][1] =  9;board_possible_move[107][2] = 10;
        board_possible_move[108][0] = 11;board_possible_move[108][1] = 12;board_possible_move[108][2] = 13;
        board_possible_move[109][0] = 14;board_possible_move[109][1] = 15;board_possible_move[109][2] = 16;
        board_possible_move[110][0] = 17;board_possible_move[110][1] = 18;board_possible_move[110][2] = 19;
        board_possible_move[111][0] = 20;board_possible_move[111][1] = 21;board_possible_move[111][2] = -1;

        board_possible_move[96][0]  = -1;board_possible_move[96][1]  = 22;board_possible_move[96][2]  = 23;
        board_possible_move[97][0]  = 24;board_possible_move[97][1]  = 25;board_possible_move[97][2]  = 26;
        board_possible_move[98][0]  = 27;board_possible_move[98][1]  = 28;board_possible_move[98][2]  = 29;
        board_possible_move[99][0]  = 30;board_possible_move[99][1]  = 31;board_possible_move[99][2]  = 32;
        board_possible_move[100][0] = 33;board_possible_move[100][1] = 34;board_possible_move[100][2] = 35;
        board_possible_move[101][0] = 36;board_possible_move[101][1] = 37;board_possible_move[101][2] = 38;
        board_possible_move[102][0] = 39;board_possible_move[102][1] = 40;board_possible_move[102][2] = 41;
        board_possible_move[103][0] = 42;board_possible_move[103][1] = 43;board_possible_move[103][2] = -1;

        board_possible_move[88][0] = -1;board_possible_move[88][1] = 44;board_possible_move[88][2] = 45;
        board_possible_move[89][0] = 46;board_possible_move[89][1] = 47;board_possible_move[89][2] = 48;
        board_possible_move[90][0] = 49;board_possible_move[90][1] = 50;board_possible_move[90][2] = 51;
        board_possible_move[91][0] = 52;board_possible_move[91][1] = 53;board_possible_move[91][2] = 54;
        board_possible_move[92][0] = 55;board_possible_move[92][1] = 56;board_possible_move[92][2] = 57;
        board_possible_move[93][0] = 58;board_possible_move[93][1] = 59;board_possible_move[93][2] = 60;
        board_possible_move[94][0] = 61;board_possible_move[94][1] = 62;board_possible_move[94][2] = 63;
        board_possible_move[95][0] = 64;board_possible_move[95][1] = 65;board_possible_move[95][2] = -1;

        board_possible_move[80][0] = -1;board_possible_move[80][1] = 66;board_possible_move[80][2] = 67;
        board_possible_move[81][0] = 68;board_possible_move[81][1] = 69;board_possible_move[81][2] = 70;
        board_possible_move[82][0] = 71;board_possible_move[82][1] = 72;board_possible_move[82][2] = 73;
        board_possible_move[83][0] = 74;board_possible_move[83][1] = 75;board_possible_move[83][2] = 76;
        board_possible_move[84][0] = 77;board_possible_move[84][1] = 78;board_possible_move[84][2] = 79;
        board_possible_move[85][0] = 80;board_possible_move[85][1] = 81;board_possible_move[85][2] = 82;
        board_possible_move[86][0] = 83;board_possible_move[86][1] = 84;board_possible_move[86][2] = 85;
        board_possible_move[87][0] = 86;board_possible_move[87][1] = 87;board_possible_move[87][2] = -1;

        board_possible_move[72][0] =  -1;board_possible_move[72][1] =  88;board_possible_move[72][2] =  89;
        board_possible_move[73][0] =  90;board_possible_move[73][1] =  91;board_possible_move[73][2] =  92;
        board_possible_move[74][0] =  93;board_possible_move[74][1] =  94;board_possible_move[74][2] =  95;
        board_possible_move[75][0] =  96;board_possible_move[75][1] =  97;board_possible_move[75][2] =  98;
        board_possible_move[76][0] =  99;board_possible_move[76][1] = 100;board_possible_move[76][2] = 101;
        board_possible_move[77][0] = 102;board_possible_move[77][1] = 103;board_possible_move[77][2] = 104;
        board_possible_move[78][0] = 105;board_possible_move[78][1] = 106;board_possible_move[78][2] = 107;
        board_possible_move[79][0] = 108;board_possible_move[79][1] = 109;board_possible_move[79][2] =  -1;

        board_possible_move[64][0] =  -1;board_possible_move[64][1] = 110;board_possible_move[64][2] = 111;
        board_possible_move[65][0] = 112;board_possible_move[65][1] = 113;board_possible_move[65][2] = 114;
        board_possible_move[66][0] = 115;board_possible_move[66][1] = 116;board_possible_move[66][2] = 117;
        board_possible_move[67][0] = 118;board_possible_move[67][1] = 119;board_possible_move[67][2] = 120;
        board_possible_move[68][0] = 121;board_possible_move[68][1] = 122;board_possible_move[68][2] = 123;
        board_possible_move[69][0] = 124;board_possible_move[69][1] = 125;board_possible_move[69][2] = 126;
        board_possible_move[70][0] = 127;board_possible_move[70][1] = 128;board_possible_move[70][2] = 129;
        board_possible_move[71][0] = 130;board_possible_move[71][1] = 131;board_possible_move[71][2] =  -1;

        board_possible_move[56][0] =  -1;board_possible_move[56][1] = 132;board_possible_move[56][2] = 133;
        board_possible_move[57][0] = 134;board_possible_move[57][1] = 135;board_possible_move[57][2] = 136;
        board_possible_move[58][0] = 137;board_possible_move[58][1] = 138;board_possible_move[58][2] = 139;
        board_possible_move[59][0] = 140;board_possible_move[59][1] = 141;board_possible_move[59][2] = 142;
        board_possible_move[60][0] = 143;board_possible_move[60][1] = 144;board_possible_move[60][2] = 145;
        board_possible_move[61][0] = 146;board_possible_move[61][1] = 147;board_possible_move[61][2] = 148;
        board_possible_move[62][0] = 149;board_possible_move[62][1] = 150;board_possible_move[62][2] = 151;
        board_possible_move[63][0] = 152;board_possible_move[63][1] = 153;board_possible_move[63][2] =  -1;
        //white
        //                          left                           middle                            right
        board_possible_move[0][0] =  -1;board_possible_move[0][1] = 154;board_possible_move[0][2] = 155;
        board_possible_move[1][0] = 156;board_possible_move[1][1] = 157;board_possible_move[1][2] = 158;
        board_possible_move[2][0] = 159;board_possible_move[2][1] = 160;board_possible_move[2][2] = 161;
        board_possible_move[3][0] = 162;board_possible_move[3][1] = 163;board_possible_move[3][2] = 164;
        board_possible_move[4][0] = 165;board_possible_move[4][1] = 166;board_possible_move[4][2] = 167;
        board_possible_move[5][0] = 168;board_possible_move[5][1] = 169;board_possible_move[5][2] = 170;
        board_possible_move[6][0] = 171;board_possible_move[6][1] = 172;board_possible_move[6][2] = 173;
        board_possible_move[7][0] = 174;board_possible_move[7][1] = 175;board_possible_move[7][2] =  -1;

        board_possible_move[8][0]  =  -1;board_possible_move[8][1]  = 176;board_possible_move[8][2]  = 177;
        board_possible_move[9][0]  = 178;board_possible_move[9][1]  = 179;board_possible_move[9][2]  = 180;
        board_possible_move[10][0] = 181;board_possible_move[10][1] = 182;board_possible_move[10][2] = 183;
        board_possible_move[11][0] = 184;board_possible_move[11][1] = 185;board_possible_move[11][2] = 186;
        board_possible_move[12][0] = 187;board_possible_move[12][1] = 188;board_possible_move[12][2] = 189;
        board_possible_move[13][0] = 190;board_possible_move[13][1] = 191;board_possible_move[13][2] = 192;
        board_possible_move[14][0] = 193;board_possible_move[14][1] = 194;board_possible_move[14][2] = 195;
        board_possible_move[15][0] = 196;board_possible_move[15][1] = 197;board_possible_move[15][2] =  -1;

        board_possible_move[16][0] =  -1;board_possible_move[16][1] = 198;board_possible_move[16][2] = 199;
        board_possible_move[17][0] = 200;board_possible_move[17][1] = 201;board_possible_move[17][2] = 202;
        board_possible_move[18][0] = 203;board_possible_move[18][1] = 204;board_possible_move[18][2] = 205;
        board_possible_move[19][0] = 206;board_possible_move[19][1] = 207;board_possible_move[19][2] = 208;
        board_possible_move[20][0] = 209;board_possible_move[20][1] = 210;board_possible_move[20][2] = 211;
        board_possible_move[21][0] = 212;board_possible_move[21][1] = 213;board_possible_move[21][2] = 214;
        board_possible_move[22][0] = 215;board_possible_move[22][1] = 216;board_possible_move[22][2] = 217;
        board_possible_move[23][0] = 218;board_possible_move[23][1] = 219;board_possible_move[23][2] =  -1;

        board_possible_move[24][0] =  -1;board_possible_move[24][1] = 220;board_possible_move[24][2] = 221;
        board_possible_move[25][0] = 222;board_possible_move[25][1] = 223;board_possible_move[25][2] = 224;
        board_possible_move[26][0] = 225;board_possible_move[26][1] = 226;board_possible_move[26][2] = 227;
        board_possible_move[27][0] = 228;board_possible_move[27][1] = 229;board_possible_move[27][2] = 230;
        board_possible_move[28][0] = 231;board_possible_move[28][1] = 232;board_possible_move[28][2] = 233;
        board_possible_move[29][0] = 234;board_possible_move[29][1] = 235;board_possible_move[29][2] = 236;
        board_possible_move[30][0] = 237;board_possible_move[30][1] = 238;board_possible_move[30][2] = 239;
        board_possible_move[31][0] = 240;board_possible_move[31][1] = 241;board_possible_move[31][2] =  -1;

        board_possible_move[32][0] =  -1;board_possible_move[32][1] = 242;board_possible_move[32][2] = 243;
        board_possible_move[33][0] = 244;board_possible_move[33][1] = 245;board_possible_move[33][2] = 246;
        board_possible_move[34][0] = 247;board_possible_move[34][1] = 248;board_possible_move[34][2] = 249;
        board_possible_move[35][0] = 250;board_possible_move[35][1] = 251;board_possible_move[35][2] = 252;
        board_possible_move[36][0] = 253;board_possible_move[36][1] = 254;board_possible_move[36][2] = 255;
        board_possible_move[37][0] = 256;board_possible_move[37][1] = 257;board_possible_move[37][2] = 258;
        board_possible_move[38][0] = 259;board_possible_move[38][1] = 260;board_possible_move[38][2] = 261;
        board_possible_move[39][0] = 262;board_possible_move[39][1] = 263;board_possible_move[39][2] =  -1;

        board_possible_move[40][0] =  -1;board_possible_move[40][1] = 264;board_possible_move[40][2] = 265;
        board_possible_move[41][0] = 266;board_possible_move[41][1] = 267;board_possible_move[41][2] = 268;
        board_possible_move[42][0] = 269;board_possible_move[42][1] = 270;board_possible_move[42][2] = 271;
        board_possible_move[43][0] = 272;board_possible_move[43][1] = 273;board_possible_move[43][2] = 274;
        board_possible_move[44][0] = 275;board_possible_move[44][1] = 276;board_possible_move[44][2] = 277;
        board_possible_move[45][0] = 278;board_possible_move[45][1] = 279;board_possible_move[45][2] = 280;
        board_possible_move[46][0] = 281;board_possible_move[46][1] = 282;board_possible_move[46][2] = 283;
        board_possible_move[47][0] = 284;board_possible_move[47][1] = 285;board_possible_move[47][2] =  -1;

        board_possible_move[48][0] =  -1;board_possible_move[48][1] = 286;board_possible_move[48][2] = 287;
        board_possible_move[49][0] = 288;board_possible_move[49][1] = 289;board_possible_move[49][2] = 290;
        board_possible_move[50][0] = 291;board_possible_move[50][1] = 292;board_possible_move[50][2] = 293;
        board_possible_move[51][0] = 294;board_possible_move[51][1] = 295;board_possible_move[51][2] = 296;
        board_possible_move[52][0] = 297;board_possible_move[52][1] = 298;board_possible_move[52][2] = 299;
        board_possible_move[53][0] = 300;board_possible_move[53][1] = 301;board_possible_move[53][2] = 302;
        board_possible_move[54][0] = 303;board_possible_move[54][1] = 304;board_possible_move[54][2] = 305;
        board_possible_move[55][0] = 306;board_possible_move[55][1] = 307;board_possible_move[55][2] =  -1;
    }

    StatePtr BreakThroughState::clone() const { return std::make_unique<BreakThroughState>(*this); }

    void BreakThroughState::apply_action(const Action& action) {
        Player player = current_player();
        //std::cout<<action<<std::endl;
        int bpo = possible_move[action] / 100;
        int apo = possible_move[action] % 100;
        //std::cout<<bpo<<" "<<apo<<std::endl;
        board_[ bpo ] = 2;
        if(board_[apo] != 2)
            chess_num[board_[apo]]--;
        
        board_[ apo ] = player;
        ++turn_;
        if (arrive(player)) {
            winner_ = player;
        }
        history_.push_back(action);
    }
    //have some problem
    std::vector<Action> BreakThroughState::legal_actions() const {
        std::vector<Action> actions;
        Player player = current_player();
        int _move;
        std::cout<<"turn:"<<player<<std::endl;
        for(int i = 0; i < 64; i++){
            if(board_[i] == player){
                if(player == 0 && i <= 55){
                    //
                    _move = board_possible_move[i][0];
                    //std::cout<<i<<" "<<_move<<std::endl;
                    if(_move != -1 && board_[ (possible_move[_move]%100) ] != player)
                        actions.push_back(_move);
                    //
                    _move = board_possible_move[i][1];
                    //std::cout<<i<<" "<<_move<<std::endl;
                    if(board_[ (possible_move[_move]%100) ] == 2)
                        actions.push_back(_move);
                    //
                    _move = board_possible_move[i][2];
                    //std::cout<<i<<" "<<_move<<std::endl;
                    if(_move != -1 && board_[ (possible_move[_move]%100) ] != player)
                        actions.push_back(_move);
                }else if(player == 1 && i >= 8){
                    //
                    _move = board_possible_move[i+48][0];
                    if(_move != -1 && board_[ (possible_move[_move]%100) ] != player)
                        actions.push_back(_move);
                    //
                    _move = board_possible_move[i+48][1];
                    if(board_[ (possible_move[_move]%100) ] == 2)
                        actions.push_back(_move);
                    //
                    _move = board_possible_move[i+48][2];
                    if(_move != -1 && board_[ (possible_move[_move]%100) ] != player)
                        actions.push_back(_move);
                }
            }
        }
        /*
        for (int i = 0; i < 64; i++) {
            if (board_[i] == player) {
                if(player == 1){
                    if((i-8)>=0 && board_[i-8] == 2){//board_[i-8] = 2
                        //before_m[act] = i;
                        //after_m[act] = i-8;
                        act = i * 100 + (i-8);
                        actions.push_back(act);
                    }
                    //
                    if(i % 8 == 0){
                        if((i-7)>=0 && board_[i-7] != 1){
                            //before_m[act] = i;
                            //after_m[act] = i-7;
                            act = i * 100 + (i-7);
                            actions.push_back(act);
                        }
                    }else if(i % 8 == 7){
                        if((i-9)>=0 && board_[i-9] != 1){
                            //before_m[act] = i;
                            //after_m[act] = i-9;
                            act = i * 100 + (i-9);
                            actions.push_back(act);
                        }
                    }else{
                        if((i-9)>=0 && board_[i-9] != 1){//board_[i-9] = 0 or 2
                            //before_m[act] = i;
                            //after_m[act] = i-9;
                            act = i * 100 + (i-9);
                            actions.push_back(act);
                        }
                        if((i-7)>=0 && board_[i-7] != 1){//board_[i-7] = 0 or 2
                            //before_m[act] = i;
                            //after_m[act] = i-7;
                            act = i * 100 + (i-7);
                            actions.push_back(act);
                        }
                    }
                }else if(player == 0){
                    if((i+8)<=63 && board_[i+8] == 2){
                            //before_m[act] = i;
                            //after_m[act] = i+8;
                            act = i * 100 + (i+8);
                            actions.push_back(act);
                    }
                    //
                    if(i % 8 == 0){
                        if((i+9)<=63 && board_[i+9] != 0){
                            //before_m[act] = i;
                            //after_m[act] = i+9;
                            act = i * 100 + (i+9);
                            actions.push_back(act);
                        }
                    }else if(i % 8 == 7){
                        if((i+7)<=63 &&board_[i+7] != 0){
                            //before_m[act] = i;
                            //after_m[act] = i+7;
                            act = i * 100 + (i+7);
                            actions.push_back(act);
                        }
                    }else{
                        if((i+9)<=63 &&board_[i+9] != 0){
                            //before_m[act] = i;
                            //after_m[act] = i+9;
                            act = i * 100 + (i+9);
                            actions.push_back(act);
                        }
                        if((i+7)<=63 &&board_[i+7] != 0){
                            //before_m[act] = i;
                            //after_m[act] = i+7;
                            act = i * 100 + (i+7);
                            actions.push_back(act);
                        }
                    }
                }
                //actions.push_back(i);
            }
        }*/
        
        return actions;
    }
    
    std::string BreakThroughState::to_string() const {
        std::string c;
        for (const auto& b : board_) c += "WB "[b];
        std::stringstream ss;
        ss << "┌───┬───┬───┬───┬───┬───┬───┬───┐" << std::endl;
        ss << "│ " << c[0] << " │ " << c[1] << " │ " << c[2] << " │ " << c[3] << " | " << c[4] << " | " << c[5] << " | " << c[6] << " | " << c[7] << " |" << std::endl;
        ss << "├───┼───┼───┼───┼───┼───┼───┼───┤" << std::endl;
        ss << "│ " << c[8] << " │ " << c[9] << " │ " << c[10] << " │ " << c[11] << " | " << c[12] << " | " << c[13] << " | " << c[14] << " | " << c[15] << " |" << std::endl;
        ss << "├───┼───┼───┼───┼───┼───┼───┼───┤" << std::endl;
        ss << "│ " << c[16] << " │ " << c[17] << " │ " << c[18] << " │ " << c[19] << " | " << c[20] << " | " << c[21] << " | " << c[22] << " | " << c[23] << " |" << std::endl;
        ss << "├───┼───┼───┼───┼───┼───┼───┼───┤" << std::endl;
        ss << "│ " << c[24] << " │ " << c[25] << " │ " << c[26] << " │ " << c[27] << " | " << c[28] << " | " << c[29] << " | " << c[30] << " | " << c[31] << " |" << std::endl;
        ss << "├───┼───┼───┼───┼───┼───┼───┼───┤" << std::endl;
        ss << "│ " << c[32] << " │ " << c[33] << " │ " << c[34] << " │ " << c[35] << " | " << c[36] << " | " << c[37] << " | " << c[38] << " | " << c[39] << " |" << std::endl;
        ss << "├───┼───┼───┼───┼───┼───┼───┼───┤" << std::endl;
        ss << "│ " << c[40] << " │ " << c[41] << " │ " << c[42] << " │ " << c[43] << " | " << c[44] << " | " << c[45] << " | " << c[46] << " | " << c[47] << " |" << std::endl;
        ss << "├───┼───┼───┼───┼───┼───┼───┼───┤" << std::endl;
        ss << "│ " << c[48] << " │ " << c[49] << " │ " << c[50] << " │ " << c[51] << " | " << c[52] << " | " << c[53] << " | " << c[54] << " | " << c[55] << " |" << std::endl;
        ss << "├───┼───┼───┼───┼───┼───┼───┼───┤" << std::endl;
        ss << "│ " << c[56] << " │ " << c[57] << " │ " << c[58] << " │ " << c[59] << " | " << c[60] << " | " << c[61] << " | " << c[62] << " | " << c[63] << " |" << std::endl;
        ss << "└───┴───┴───┴───┴───┴───┴───┴───┘" << std::endl;
        return ss.str();
    }

    bool BreakThroughState::is_terminal() const { return turn_ == 92 || winner_ != -1; }

    Player BreakThroughState::current_player() const { return ((turn_ % 2 + 1) % 2); }
    //use for?
    std::vector<float> BreakThroughState::observation_tensor() const {
        std::vector<float> tensor;
        auto shape = game()->observation_tensor_shape();
        auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
        tensor.reserve(size);

        for (int i = 0; i < 64; ++i) {
            tensor.push_back(static_cast<float>(board_[i] == 0));
        }
        for (int i = 0; i < 64; ++i) {
            tensor.push_back(static_cast<float>(board_[i] == 1));
        }
        for (int i = 0; i < 64; ++i) {
            tensor.push_back(static_cast<float>(board_[i] == 2));
        }
        Player player = current_player();
        for (int i = 0; i < 64; ++i) {
            tensor.push_back(static_cast<float>(player));
        }

        return tensor;
    }

    bool BreakThroughState::arrive(const Player& player) const {
        int b = player;
        if(chess_num[(player+1)%2]==0)
            return true;
        if(b == 1)//black player
            return (board_[0] == b || board_[1] == b || board_[2] == b || board_[3] == b ||
                    board_[4] == b || board_[5] == b || board_[6] == b || board_[7] == b);
        else if(b==0)//white player
            return (board_[56] == b || board_[57] == b || board_[58] == b || board_[59] == b ||
                    board_[60] == b || board_[61] == b || board_[62] == b || board_[63] == b);

        return false; 
    }

    std::string BreakThroughState::serialize() const {
        std::stringstream ss;
        for (const Action action : history_) {
            ss << action << " ";
        }
        return ss.str();
    }
    
    std::vector<float> BreakThroughState::rewards() const {
        if (winner_ == -1) {
            return {0.0F, 0.0F};
        }
        if (winner_ == 0) {
            return {1.0F, -1.0F};
        }
        // if (winner_ == 1)
        return {-1.0F, 1.0F};
    }


    std::string BreakThroughGame::name() const { return "breakthrough"; } 
    int BreakThroughGame::num_players() const { return 2; }
    int BreakThroughGame::num_distinct_actions() const { return 308; }
    //use for?
    std::vector<int> BreakThroughGame::observation_tensor_shape() const { return {4, 8, 8}; }
    
    StatePtr BreakThroughGame::new_initial_state() const {
        return std::make_unique<BreakThroughState>(shared_from_this());
    }
    //??
    int BreakThroughGame::num_transformations() const { return 8; }
    //??
    std::vector<float> BreakThroughGame::transform_observation(const std::vector<float>& observation,
                                                            int type) const {
    std::vector<float> transformed_observation(observation);
        //transform(&transformed_observation[0], type);
        //transform(&transformed_observation[9], type);
        //transform(&transformed_observation[18], type);
        return transformed_observation;
    }

    std::vector<float> BreakThroughGame::transform_policy(const std::vector<float>& policy,
                                                   int type) const {
        std::vector<float> transformed_policy(policy);
        //transform(&transformed_policy[0], type);
        return transformed_policy;
    }

    std::vector<float> BreakThroughGame::restore_policy(const std::vector<float>& policy,
                                                 int type) const {
        std::vector<float> restored_policy(policy);
        int restored_type;
        switch (type) {
            default:
            case 0:
                restored_type = 0;
                break;
            case 1:
                restored_type = 3;
                break;
            case 2:
                restored_type = 2;
                break;
            case 3:
                restored_type = 1;
                break;
            case 4:
                restored_type = 4;
                break;
            case 5:
                restored_type = 5;
                break;
            case 6:
                restored_type = 6;
                break;
            case 7:
                restored_type = 7;
                break;
        }
        //transform(&restored_policy[0], restored_type);
        return restored_policy;
    }

    StatePtr BreakThroughGame::deserialize_state(const std::string& str) const {
        std::stringstream ss(str);
        int action;
        StatePtr state = new_initial_state();
        //ss>>action;
        //std::cout<<action<<std::endl;
        while (ss >> action) {
            //std::cout<<"act\n";
            state->apply_action(action);
        }
        return state->clone();
    }
}
