/* tensor_test.cpp */

/*
 * Test of different member functions of the tensor classes in Tensor.hpp.
 *
 * g++ -g -Wextra -Wall tensor_test.cpp -o tensor_test.exe
 * tensor_test.exe
 */

#include "../../Tensor.hpp"
#include "../unit_test.hpp"



/**
 * Unit tests.
 */
int main() {
    START_TEST("Tensor")


    /* 1. Test RANK=1 constructor and assignment */
    std::cout << "Starting Test: RANK=1 constructor and assignment\n";
    jai::Tensor<1> t1_1(3);
    t1_1[0] = 1; t1_1[1] = 2; t1_1[2] = -1;

    test_equals( t1_1.rank(), 1 );
    test_equals( t1_1.totalSize(), 3 );
    test_equals( t1_1.size(), 3 );
    test_equals( t1_1[0], 1 );
    test_equals( t1_1[1], 2 );
    test_equals( t1_1[2], -1 );

    test_throws(({ jai::Tensor<1> t1_1_t1(0); }));


    /* 2. Test RANK=1 fill constructor */
    std::cout << "Starting Test: RANK=1 fill constructor\n";
    const jai::Tensor<1> t1_2(5, 10);

    test_equals( t1_2.rank(), 1 );
    test_equals( t1_2.totalSize(), 5 );
    test_equals( t1_2.size(), 5 );
    test_equals( t1_2[0], 10 );
    test_equals( t1_2[4], 10 );

    test_throws(({ jai::Tensor<1> t1_2_t1(0, 10); }));


    /* 3. Test RANK=1 element initializer constructor */
    std::cout << "Starting Test: RANK=1 element initializer constructor\n";
    const jai::Tensor<1> t1_3 = {1, 2, 3, 4, 5, 6, 7, 8};

    test_equals( t1_3.rank(), 1 );
    test_equals( t1_3.totalSize(), 8 );
    test_equals( t1_3.size(), 8 );
    test_equals( t1_3[0], 1 );
    test_equals( t1_3[3], 4 );
    test_equals( t1_3[7], 8 );


    /* 4. Test RANK=2 constructor and assignment */
    std::cout << "Starting Test: RANK=2 constructor and assignment\n";
    jai::Tensor<2> t2_4({3, 2});
    t2_4[{0, 0}] = 1; t2_4[{0, 1}] = 2; t2_4[{1, 0}] = 3; t2_4[{1,1}] = 4; t2_4[{2, 0}] = 5; t2_4[{2, 1}] = 6;

    test_equals( t2_4.rank(), 2 );
    test_equals( t2_4.totalSize(), 6 );
    test_equals( t2_4.size(0), 3 );
    test_equals( t2_4.size(1), 2 );
    test_equals( (t2_4[{0, 0}]), 1 );
    test_equals( (t2_4[{0, 1}]), 2 );
    test_equals( (t2_4[{1, 0}]), 3 );
    test_equals( (t2_4[{1, 1}]), 4 );
    test_equals( (t2_4[{2, 0}]), 5 );
    test_equals( (t2_4[{2, 1}]), 6 );
    test_equals( (t2_4[0][0]), 1 );
    test_equals( (t2_4[0][1]), 2 );
    test_equals( (t2_4[1][0]), 3 );
    test_equals( (t2_4[1][1]), 4 );
    test_equals( (t2_4[2][0]), 5 );
    test_equals( (t2_4[2][1]), 6 );

    test_throws(({ jai::Tensor<2> t2_4_t1({0, 10}); }));
    test_throws(({ jai::Tensor<2> t2_4_t2({10, 0}); }));


    /* 5. Test RANK=2 fill constructor */
    std::cout << "Starting Test: RANK=2 fill constructor\n";
    const jai::Tensor<2> t2_5({2, 3}, 50);

    test_equals( t2_5.rank(), 2 );
    test_equals( t2_5.totalSize(), 6 );
    test_equals( t2_5.size(0), 2 );
    test_equals( t2_5.size(1), 3 );
    test_equals( (t2_5[{0, 0}]), 50 );
    test_equals( (t2_5[{0, 1}]), 50 );
    test_equals( (t2_5[{0, 2}]), 50 );
    test_equals( (t2_5[{1, 0}]), 50 );
    test_equals( (t2_5[{1, 1}]), 50 );
    test_equals( (t2_5[{1, 2}]), 50 );
    test_equals( (t2_5[0][0]), 50 );
    test_equals( (t2_5[0][1]), 50 );
    test_equals( (t2_5[0][2]), 50 );
    test_equals( (t2_5[1][0]), 50 );
    test_equals( (t2_5[1][1]), 50 );
    test_equals( (t2_5[1][2]), 50 );

    test_throws(({ jai::Tensor<2> t2_4_t1({0, 10}, 51); }));
    test_throws(({ jai::Tensor<2> t2_4_t2({10, 0}, 52); }));


    /* 6. Test RANK=2 element initializer constructor */
    std::cout << "Starting Test: RANK=2 element initializer constructor\n";
    const jai::Tensor<2> t2_6 = {{61, 62, 63}, {64, 65, 66}};

    test_equals( t2_6.rank(), 2 );
    test_equals( t2_6.totalSize(), 6 );
    test_equals( t2_6.size(0), 2 );
    test_equals( t2_6.size(1), 3 );
    test_equals( (t2_6[{0, 0}]), 61 );
    test_equals( (t2_6[{0, 1}]), 62 );
    test_equals( (t2_6[{0, 2}]), 63 );
    test_equals( (t2_6[{1, 0}]), 64 );
    test_equals( (t2_6[{1, 1}]), 65 );
    test_equals( (t2_6[{1, 2}]), 66 );
    test_equals( (t2_6[0][0]), 61 );
    test_equals( (t2_6[0][1]), 62 );
    test_equals( (t2_6[0][2]), 63 );
    test_equals( (t2_6[1][0]), 64 );
    test_equals( (t2_6[1][1]), 65 );
    test_equals( (t2_6[1][2]), 66 );

    test_throws(({ jai::Tensor<2> t2_6_t1 = {{61, 62, 63}, {64, 65}}; }));
    test_throws(({ jai::Tensor<2> t2_6_t2 = {{61, 62}, {64, 65, 66}, {67, 68, 69, 70}}; }));
    test_throws(({ jai::Tensor<2> t2_6_t3 = {{61, 62, 63}, {64, 65, 66}, {}}; }));


    /* 7. Test RANK=2 Tensor element initializer constructor */
    std::cout << "Starting Test: RANK=2 Tensor element initializer constructor\n";
    const jai::Tensor<1> t1_7_1 = {71, 72};
    const jai::Tensor<1> t1_7_2 = {73, 74};
    const jai::Tensor<1> t1_7_3 = {75, 76, 77};
    const jai::Tensor<2> t2_7 = {t1_7_1, t1_7_2};

    test_equals( t2_7.rank(), 2 );
    test_equals( t2_7.totalSize(), 4 );
    test_equals( t2_7.size(0), 2 );
    test_equals( t2_7.size(1), 2 );
    test_equals( (t2_7[{0, 0}]), 71);
    test_equals( (t2_7[{0, 1}]), 72 );
    test_equals( (t2_7[{1, 0}]), 73 );
    test_equals( (t2_7[{1, 1}]), 74 );
    test_equals( (t2_7[0][0]), 71 );
    test_equals( (t2_7[0][1]), 72 );
    test_equals( (t2_7[1][0]), 73 );
    test_equals( (t2_7[1][1]), 74 );

    test_throws(({ jai::Tensor<2> t2_7_t1 = {t1_7_1, t1_7_2, t1_7_3}; }));


    /* 8. Test inner tensor accessor */
    std::cout << "Starting Test: inner tensor accessor\n";
    jai::Tensor<3> t3_8({3, 2, 2}, 81);
    jai::VTensor<2> vt2_8 = t3_8[1];
    vt2_8[{0, 0}] = 82; vt2_8[{1, 0}] = 83; vt2_8[{0, 1}] = 84;

    test_equals( vt2_8.totalSize(), 4 );
    test_equals( vt2_8.size(0), 2 );
    test_equals( vt2_8.size(1), 2 );
    test_equals( (t3_8[{1, 0, 0}]), 82 );
    test_equals( (t3_8[{1, 1, 0}]), 83 );
    test_equals( (t3_8[{1, 0, 1}]), 84 );
    test_equals( (vt2_8[{0, 0}]), 82 );
    test_equals( (vt2_8[{1, 0}]), 83 );
    test_equals( (vt2_8[{0, 1}]), 84 );


    /* 9. Test view() */
    std::cout << "Starting Test: view()\n";
    jai::Tensor<1> t1_9(8, 90);
    jai::VTensor<1> vt1_9 = t1_9.view();
    vt1_9[4] = 91;
    t1_9[5] = 92;
    jai::Tensor<2> t2_9({100, 50}, 93);
    jai::VTensor<2> vt2_9 = t2_9.view();
    vt2_9[{1, 1}] = 94;
    t2_9[{2, 2}] = 95;

    test_equals( vt1_9.totalSize(), 8 );
    test_equals( vt1_9.size(), 8 );
    test_equals( t1_9[0], 90 );
    test_equals( t1_9[4], 91 );
    test_equals( t1_9[5], 92 );
    test_equals( t1_9[7], 90 );
    test_equals( vt1_9[0], 90 );
    test_equals( vt1_9[4], 91 );
    test_equals( vt1_9[5], 92 );
    test_equals( vt1_9[7], 90 );

    test_equals( vt2_9.rank(), 2 );
    test_equals( vt2_9.totalSize(), 5000 );
    test_equals( vt2_9.size(0), 100 );
    test_equals( vt2_9.size(1), 50 );
    test_equals( (t2_9[{0, 0}]), 93 );
    test_equals( (t2_9[{1, 1}]), 94 );
    test_equals( (t2_9[{2, 2}]), 95 );
    test_equals( (t2_9[{99, 49}]), 93 );
    test_equals( (vt2_9[{0, 0}]), 93 );
    test_equals( (vt2_9[{1, 1}]), 94 );
    test_equals( (vt2_9[{2, 2}]), 95 );
    test_equals( (vt2_9[{99, 49}]), 93 );


    /* 10. Test rankUp() */
    std::cout << "Starting Test: rankUp()\n";
    const jai::Tensor<1> t1_10 = {101, 102, 103};
    const jai::VTensor<2> vt2_10 = t1_10.rankUp();

    test_equals( vt2_10.totalSize(), 3 );
    test_equals( vt2_10.size(0), 3 );
    test_equals( vt2_10.size(1), 1 );
    test_equals( (vt2_10[{0, 0}]), 101 );
    test_equals( (vt2_10[{1, 0}]), 102 );
    test_equals( (vt2_10[{2, 0}]), 103 );


    /* 11. Test flattened() */
    std::cout << "Starting Test: flattened()\n";
    jai::Tensor<3> t3_11 = {
        { {1101, 1102, 1103}, {1104, 1105, 1106} },
        { {1107, 1108, 1109}, {1110, 1111, 1112} },
        { {1113, 1114, 1115}, {1116, 1117, 1118} }
    };
    jai::VTensor<1> vt1_11 = t3_11.flattened();
    vt1_11[10] = 1120;

    test_equals( vt1_11.totalSize(), 18 );
    test_equals( vt1_11.size(), 18 );
    test_equals( (vt1_11[0]), 1101 );
    test_equals( (vt1_11[1]), 1102 );
    test_equals( (vt1_11[2]), 1103 );
    test_equals( (vt1_11[3]), 1104 );
    test_equals( (vt1_11[4]), 1105 );
    test_equals( (vt1_11[5]), 1106 );
    test_equals( (vt1_11[10]), 1120 );
    test_equals( (vt1_11[12]), 1113 );
    test_equals( (vt1_11[13]), 1114 );
    test_equals( (vt1_11[14]), 1115 );
    test_equals( (vt1_11[15]), 1116 );
    test_equals( (vt1_11[16]), 1117 );
    test_equals( (vt1_11[17]), 1118 );


    /* 12. Test addition operator */
    std::cout << "Starting Test: addition operator\n";
    const jai::Tensor<1> t1_12_1 = {121, 122, 123, 124};
    const jai::Tensor<1> t1_12_2 = {125, 126, 127, 128};
    const jai::Tensor<1> t1_12_3 = t1_12_1 + t1_12_2;

    test_equals( t1_12_3.totalSize(), 4 );
    test_equals( t1_12_3.size(), 4 );
    test_equals( (t1_12_3[0]), 246 );
    test_equals( (t1_12_3[1]), 248 );
    test_equals( (t1_12_3[2]), 250 );
    test_equals( (t1_12_3[3]), 252 );


    /* 13. Test subtraction operator */
    std::cout << "Starting Test: subtraction operator\n";
    const jai::Tensor<1> t1_13_1 = {131, 132, 133, 134, 135};
    const jai::Tensor<1> t1_13_2 = {139, 138, 136, 135, 134};
    const jai::Tensor<1> t1_13_3 = t1_13_1 - t1_13_2;

    test_equals( t1_13_3.totalSize(), 5 );
    test_equals( t1_13_3.size(), 5 );
    test_equals( (t1_13_3[0]), -8 );
    test_equals( (t1_13_3[1]), -6 );
    test_equals( (t1_13_3[2]), -3 );
    test_equals( (t1_13_3[3]), -1 );
    test_equals( (t1_13_3[4]), 1 );


    /* 14. Test multiplication operator */
    std::cout << "Starting Test: multiplication operator\n";
    const jai::Tensor<2> t2_14_1 = {{141, 142}, {143, 144}};
    const jai::Tensor<2> t2_14_2 = t2_14_1 * 10;
    const jai::Tensor<2> t2_14_3 = 100 * t2_14_1;

    test_equals( t2_14_2.totalSize(), 4 );
    test_equals( t2_14_2.size(0), 2 );
    test_equals( t2_14_2.size(1), 2 );
    test_equals( (t2_14_2[{0,0}]), 1410 );
    test_equals( (t2_14_2[{0,1}]), 1420 );
    test_equals( (t2_14_2[{1,0}]), 1430 );
    test_equals( (t2_14_2[{1,1}]), 1440 );

    test_equals( t2_14_3.totalSize(), 4 );
    test_equals( t2_14_3.size(0), 2 );
    test_equals( t2_14_3.size(1), 2 );
    test_equals( (t2_14_3[{0,0}]), 14100 );
    test_equals( (t2_14_3[{0,1}]), 14200 );
    test_equals( (t2_14_3[{1,0}]), 14300 );
    test_equals( (t2_14_3[{1,1}]), 14400 );


    /* 15. Test division operator */
    std::cout << "Starting Test: division operator\n";
    const jai::Tensor<2> t2_15_1 = {{1510, 1520}, {1530, 1540}};
    const jai::Tensor<2> t2_15_2 = t2_15_1 / 10;

    test_equals( t2_15_2.totalSize(), 4 );
    test_equals( t2_15_2.size(0), 2 );
    test_equals( t2_15_2.size(1), 2 );
    test_equals( (t2_15_2[{0,0}]), 151);
    test_equals( (t2_15_2[{0,1}]), 152);
    test_equals( (t2_15_2[{1,0}]), 153);
    test_equals( (t2_15_2[{1,1}]), 154);


    /* 16. Test set() */
    std::cout << "Starting test: set()\n";
    jai::Tensor<2> t2_16_1 = {{1601, 1602, 1603}, {1604, 1605, 1606}};
    const jai::Tensor<2> t2_16_2 = {{1607, 1608, 1609}, {1610, 1611, 1612}};
    const jai::Tensor<1> t1_16_1 = {1613, 1614, 1615};

    test_equals( (t2_16_1[{0, 0}]), 1601);
    test_equals( (t2_16_1[{0, 1}]), 1602);
    test_equals( (t2_16_1[{0, 2}]), 1603);
    test_equals( (t2_16_1[{1, 0}]), 1604);
    test_equals( (t2_16_1[{1, 1}]), 1605);
    test_equals( (t2_16_1[{1, 2}]), 1606);

    t2_16_1.set(t2_16_2);

    test_equals( (t2_16_1[{0, 0}]), 1607);
    test_equals( (t2_16_1[{0, 1}]), 1608);
    test_equals( (t2_16_1[{0, 2}]), 1609);
    test_equals( (t2_16_1[{1, 0}]), 1610);
    test_equals( (t2_16_1[{1, 1}]), 1611);
    test_equals( (t2_16_1[{1, 2}]), 1612);

    t2_16_1[0].set(t1_16_1);

    test_equals( (t2_16_1[{0, 0}]), 1613);
    test_equals( (t2_16_1[{0, 1}]), 1614);
    test_equals( (t2_16_1[{0, 2}]), 1615);
    test_equals( (t2_16_1[{1, 0}]), 1610);
    test_equals( (t2_16_1[{1, 1}]), 1611);
    test_equals( (t2_16_1[{1, 2}]), 1612);


    /* 17. Test addTo() */
    std::cout << "Starting test: addTo()\n";
    jai::Tensor<1> t1_17_1 = {171, 172, 173};
    const jai::Tensor<1> t1_17_2 = {174, 175, 176};

    t1_17_1.addTo(t1_17_2);

    test_equals( t1_17_1[0], 345 );
    test_equals( t1_17_1[1], 347 );
    test_equals( t1_17_1[2], 349 );


    /* 18. Test subFrom() */
    std::cout << "Starting test: subFrom()\n";
    jai::Tensor<1> t1_18_1 = {181, 182, 183};
    const jai::Tensor<1> t1_18_2 = {184, 185, 186};

    t1_18_1.subFrom(t1_18_2);

    test_equals( t1_18_1[0], -3 );
    test_equals( t1_18_1[1], -3 );
    test_equals( t1_18_1[2], -3 );


    /* 19. Test scaleBy() */
    std::cout << "Starting test: scaleBy()\n";
    jai::Tensor<1> t1_19_1 = {191, 192, 193};

    t1_19_1.scaleBy(2);

    test_equals( t1_19_1[0], 382 );
    test_equals( t1_19_1[1], 384 );
    test_equals( t1_19_1[2], 386 );


    /* 20. Test mag() */
    std::cout << "Starting test: mag()\n";
    const jai::Tensor<1> t1_20_1 = {201, -202, 203};
    const jai::Tensor<1> t1_20_2 = {0, 0, 204};
    const jai::Tensor<1> t1_20_3 = {0, -205, 0};

    test_float_equals( t1_20_1.mag(), 349.877121287 );
    test_float_equals( t1_20_2.mag(), 204 );
    test_float_equals( t1_20_3.mag(), 205 );

    /* 21. Test squaredMag() */
    std::cout << "Starting test: squaredMag()\n";
    const jai::Tensor<1> t1_21_1 = {211, -212, 213};
    const jai::Tensor<1> t1_21_2 = {0, 0, 214};
    const jai::Tensor<1> t1_21_3 = {0, -215, 0};

    test_float_equals( t1_21_1.squaredMag(), 134834 );
    test_float_equals( t1_21_2.squaredMag(), 45796 );
    test_float_equals( t1_21_3.squaredMag(), 46225 );

    /* 21. Test dot() */
    std::cout << "Starting test: dot()\n";
    const jai::Tensor<1> t1_22_1 = {211, 212, -213, 214};
    const jai::Tensor<1> t1_22_2 = {215, -216, 217, 218};

    test_float_equals( t1_22_1.dot(t1_22_2), 4 );

    /* 23. Test isSameSize() */
    std::cout << "Starting test: isSameSize()\n";
    const jai::Tensor<3> t3_23_1 = { { {2301, 2302}, {2303, 2304}, {2305, 2306} }, { {2307, 2308}, {2309, 2310}, {2311, 2312} } };
    const jai::Tensor<3> t3_23_2 = { { {2313, 2314}, {2315, 2316}, {2317, 2318} }, { {2319, 2320}, {2321, 2322}, {2323, 2324} } };
    const jai::Tensor<1> t1_23_1 = { 2325, 2326, 2327 };
    const jai::Tensor<1> t1_23_2 = { 2328, 2329, 2330 };
    const jai::Tensor<3> t3_23_3 = { { {2331, 2332}, {2333, 2334} }, { {2337, 2338}, {2339, 2340} } };
    const jai::Tensor<1> t1_23_3 = { 2343, 2344, 2345, 2346 };

    test_true( t3_23_1.isSameSize(t3_23_2) );
    test_true( t1_23_1.isSameSize(t1_23_2) );
    test_true( !t3_23_1.isSameSize(t3_23_3) );
    test_true( !t1_23_1.isSameSize(t1_23_3) );


    /* 24. Test equals operator */
    std::cout << "Starting test: equals operator\n";
    const jai::Tensor<1> t1_24_1 = { 241, 242, 244, 245 };
    const jai::Tensor<1> t1_24_2 = { 241, 242, 244, 245 };
    const jai::Tensor<1> t1_24_3 = { 241, 242, 244, 245, 246 };
    const jai::Tensor<1> t1_24_4 = { 241, 242, 244, 247 };

    test_true( t1_24_1 == t1_24_1 );
    test_true( t1_24_1 == t1_24_2 );
    test_true( !(t1_24_1 == t1_24_3) );
    test_true( !(t1_24_1 == t1_24_4) );


    /* 25. Test not equals operator */
    std::cout << "Starting test: not equals operator\n";
    const jai::Tensor<1> t1_25_1 = { 251, 252, 254, 255 };
    const jai::Tensor<1> t1_25_2 = { 251, 252, 254, 255 };
    const jai::Tensor<1> t1_25_3 = { 251, 252, 254, 255, 256 };
    const jai::Tensor<1> t1_25_4 = { 251, 252, 254, 257 };

    test_true( !(t1_25_1 != t1_25_1) );
    test_true( !(t1_25_1 != t1_25_2) );
    test_true( t1_25_1 != t1_25_3 );
    test_true( t1_25_1 != t1_25_4 );


    /* 26. Test RANK=1 transform() */
    std::cout << "Starting test: RANK=1 transform()\n";
    jai::Tensor<1> t1_26 = { 261, 262, 263, 264, 265 };

    t1_26.transform([&total_tests, &failed_tests, &t1_26](const size_t index, const float value) {
        test_equals(t1_26[index], value);

        return value * 2;
    });

    test_equals( t1_26[0], 522);
    test_equals( t1_26[1], 524);
    test_equals( t1_26[2], 526);
    test_equals( t1_26[3], 528);
    test_equals( t1_26[4], 530);


    /* 27. Test RANK=2 transform() */
    std::cout << "Starting test: RANK=2 transform()\n";
    jai::Tensor<2> t2_27 = {{271, 272, 273}, {274, 275, 276}};

    t2_27.transform([&total_tests, &failed_tests, &t2_27](const size_t indexes[2], const float value) {
        size_t _indexes[2] = {indexes[0], indexes[1]}; // For some reason `indexes` cannot be used directly
        test_equals(t2_27[_indexes], value);

        return value * 2;
    });

    test_equals( (t2_27[{0, 0}]), 542);
    test_equals( (t2_27[{0, 1}]), 544);
    test_equals( (t2_27[{0, 2}]), 546);
    test_equals( (t2_27[{1, 0}]), 548);
    test_equals( (t2_27[{1, 1}]), 550);
    test_equals( (t2_27[{1, 2}]), 552);


    /* TODO: Test cases for Tensor multiplication and division operators */


    /* TODO: Test cases for other Vector and Matrix specific operations */


    END_TEST
}