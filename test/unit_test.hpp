/* unit_test.hpp */

/*
 * This file contains utilities for unit testing.
 */



/* Text coloring start and end escape sequences for Test Failures and Test Successes */
#define TFS "\033[1;31m"
#define TFE "\033[0m"
#define TSS "\033[1;32m"
#define TSE "\033[0m"

/**
 * This should be called before any test functions are called to start the test 
 * failure counting
 * `END_TEST` should be called after this.
 */
#define START_TEST( test_name ) \
    { \
        std::cout << "Started Tests: " << test_name << "\n"; \
        int total_tests = 0; \
        int failed_tests = 0;

/**
 * This should be called after any test functions are called to end the test 
 * failure counting and display the results of all the tests
 * `START_TEST` should be called before this.
 */
#define END_TEST \
        std::cout << "Finished Tests\n"; \
        if( failed_tests > 0 ) { \
            std::cerr << TFS << failed_tests << " out of " << total_tests << " tests failed" << TFE << "\n"; \
        } else { \
            std::cout << TSS << "All tests succeeded" << TSE << "\n"; \
        } \
    }

#define test_equals( _A, _B ) \
    total_tests++; \
    try { \
        const auto A = _A; \
        const auto B = _B; \
        if( A != B ) { \
            std::cerr << TFS << "Test Equals failed at line " << __LINE__ << "; A=" << std::to_string(A) << " B=" << std::to_string(B) << TFE << "\n"; \
            failed_tests++; \
        } \
    } catch(...) { \
        std::cerr << TFS << "Test Equals threw exception at line " << __LINE__ << TFE << "\n"; \
        failed_tests++; \
    }

#define test_float_equals( _A, _B ) \
    total_tests++; \
    try { \
        const float A = _A; \
        const float B = _B; \
        if( A - 1e-6 > B || B > A + 1e-6 ) { \
            std::cerr << TFS << "Test Float Equals failed at line " << __LINE__ << "; A=" << A << " B=" << B << TFE << "\n"; \
            failed_tests++; \
        } \
    } catch(...) { \
        std::cerr << TFS << "Test Equals threw exception at line " << __LINE__ << TFE << "\n"; \
        failed_tests++; \
    }

#define test_true( _Expression ) \
    total_tests++; \
    try { \
        if( !(_Expression) ) { \
            std::cerr << TFS << "Test True failed at line " << __LINE__ << TFE << "\n"; \
            failed_tests++; \
        } \
    } catch(...) { \
        std::cerr << "Test True threw exception at line " << __LINE__ << TFE << "\n"; \
        failed_tests++; \
    }

#define test_not_throws( _Code_Block ) \
    total_tests++; \
    try { \
        _Code_Block; \
    } catch(...) { \
        std::cerr << TFS << "Assert Not Throws failed at line " << __LINE__ << TFE << "\n"; \
        failed_tests++; \
    }

#define test_throws( _Code_Block ) \
    total_tests++; \
    try { \
        _Code_Block; std::cerr << TFS << "Assert Throws failed at line " << __LINE__ << TFE << "\n"; \
        failed_tests++; \
    } catch(...) { }
