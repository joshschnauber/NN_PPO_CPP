/** tensor_test.hpp
 *  g++ -g -Wextra -Wall tensor_test.cpp -o tensor_test.exe
 */

#include "../Tensor.hpp"



/* Unit tests. 
 */
int main() {
    /* 1. Test RANK=1 constructor and assignment */
    jai::Tensor<1> t1_1(3);
    t1_1[0] = 1; t1_1[1] = 2; t1_1[2] = -1;

    assert( t1_1.rank() == 1 );
    assert( t1_1.totalSize() == 3 );
    assert( t1_1.size() == 3 );
    assert( t1_1[0] == 1 );
    assert( t1_1[1] == 2 );
    assert( t1_1[2] == -1 );


    /* 2. Test RANK=1 fill constructor */
    const jai::Tensor<1> t1_2(5, 10);
    
    assert( t1_2.rank() == 1 );
    assert( t1_2.totalSize() == 5 );
    assert( t1_2.size() == 5 );
    assert( t1_2[0] == 10 );
    assert( t1_2[4] == 10 );


    /* 3. Test RANK=1 element initializer constructor */
    const jai::Tensor<1> t1_3({1, 2, 3, 4, 5, 6, 7, 8});

    assert( t1_3.rank() == 1 );
    assert( t1_3.totalSize() == 8 );
    assert( t1_3.size() == 8 );
    assert( t1_3[0] == 1 );
    assert( t1_3[3] == 4 );
    assert( t1_3[7] == 8 );


    /* 4. Test RANK=2 constructor and assignment */
    jai::Tensor<2> t2_4({3, 2});
    t2_4[{0, 0}] = 1; t2_4[{0, 1}] = 2; t2_4[{1, 0}] = 3; t2_4[{1,1}] = 4; t2_4[{2, 0}] = 5; t2_4[{2, 1}] = 6;

    assert( t2_4.rank() == 2 );
    assert( t2_4.totalSize() == 6 );
    assert( t2_4.size(0) == 3 );
    assert( t2_4.size(1) == 2 );
    assert(( t2_4[{0, 0}] == 1 ));
    assert(( t2_4[{0, 1}] == 2 ));
    assert(( t2_4[{1, 0}] == 3 ));
    assert(( t2_4[{1, 1}] == 4 ));
    assert(( t2_4[{2, 0}] == 5 ));
    assert(( t2_4[{2, 1}] == 6 ));
    assert(( t2_4[0][0] == 1 ));
    assert(( t2_4[0][1] == 2 ));
    assert(( t2_4[1][0] == 3 ));
    assert(( t2_4[1][1] == 4 ));
    assert(( t2_4[2][0] == 5 ));
    assert(( t2_4[2][1] == 6 ));


    /* 5. Test RANK=2 fill constructor */
    const jai::Tensor<2> t2_5({2, 3}, 10);

    assert( t2_5.rank() == 2 );
    assert( t2_5.totalSize() == 6 );
    assert( t2_5.size(0) == 2 );
    assert( t2_5.size(1) == 3 );
    assert(( t2_5[{0, 0}] == 10 ));
    assert(( t2_5[{0, 1}] == 10 ));
    assert(( t2_5[{0, 2}] == 10 ));
    assert(( t2_5[{1, 0}] == 10 ));
    assert(( t2_5[{1, 1}] == 10 ));
    assert(( t2_5[{1, 2}] == 10 ));
    assert(( t2_5[0][0] == 10 ));
    assert(( t2_5[0][1] == 10 ));
    assert(( t2_5[0][2] == 10 ));
    assert(( t2_5[1][0] == 10 ));
    assert(( t2_5[1][1] == 10 ));
    assert(( t2_5[1][2] == 10 ));


    /* 6. Test RANK=2 element initializer constructor */
    const jai::Tensor<1> t1_6_1({10, 20});
    const jai::Tensor<1> t1_6_2({30, 40});
    const jai::Tensor<2> t2_6({t1_6_1, t1_6_2});

    assert( t2_6.rank() == 2 );
    assert( t2_6.totalSize() == 4 );
    assert( t2_6.size(0) == 2 );
    assert( t2_6.size(1) == 2 );
    assert(( t2_6[{0, 0}] == 10 ));
    assert(( t2_6[{0, 1}] == 20 ));
    assert(( t2_6[{1, 0}] == 30 ));
    assert(( t2_6[{1, 1}] == 40 ));
    assert(( t2_6[0][0] == 10 ));
    assert(( t2_6[0][1] == 20 ));
    assert(( t2_6[1][0] == 30 ));
    assert(( t2_6[1][1] == 40 ));


    /* 7. Test Tensor::view() */
    jai::Tensor<1> t1_7(8, 100);
    jai::VTensor<1> vt1_7 = t1_7.view();
    t1_7[5] = 101;
    jai::Tensor<2> t2_7({100, 50}, 200);
    jai::VTensor<2> vt2_7 = t2_7.view();
    t2_7[{1, 1}] = 201;

    assert( vt1_7.rank() == 1 );
    assert( vt1_7.totalSize() == 8 );
    assert( vt1_7.size() == 8 );
    assert( vt1_7[0] == 100 );
    assert( vt1_7[5] == 101 );
    assert( vt1_7[7] == 100 );

    assert( vt2_7.rank() == 1 );
    assert( vt2_7.totalSize() == 8 );
    assert( vt2_7.size(0) == 100 );
    assert( vt2_7.size(1) == 50 );
    assert(( vt2_7[{0, 0}] == 200 ));
    assert(( vt2_7[{1, 1}] == 201 ));
    assert(( vt2_7[{99, 49}] == 200 ));


    /*  */
}