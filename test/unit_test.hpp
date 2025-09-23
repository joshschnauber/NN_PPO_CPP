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
 * Starts unit testing on a larger unit of functionality, usually an entire file or class
 * `END_TESTING` should be called after this.
 */
#define START_TESTING( unit_tests_name )                                \
    {                                                                   \
        int total_unit_tests = 0;                                       \
        int total_failed_unit_tests = 0;                                \
        std::cout << "Starting Unit Tests: " << unit_tests_name << "\n";

/**
 * Finishes unit testing on a larger unit of functionality, usually an entire file or class
 * Displays the number of tests failed
 * `START_TESTING` should be called before this.
 */
#define END_TESTING                                                     \
        std::cout << "Finished Unit Tests\n";                           \
        if( total_failed_unit_tests > 0 ) {                             \
            std::cerr << TFS                                            \
                      << total_failed_unit_tests << " out of "          \
                      << total_unit_tests << " unit tests failed"       \
                      << TFE << "\n";                                   \
        } else {                                                        \
            std::cout << TSS << "All tests succeeded" << TSE << "\n";   \
        }                                                               \
    }

/**
 * Starts a unit test on a small unit of functionality, usually a function call.
 * Keeps track of the tests called.
 * `END_UNIT_TEST` should be called after this.
 */
#define UNIT_TEST( unit_test_name_ )                                    \
    {                                                                   \
        total_unit_tests++;                                             \
        const std::string unit_test_name = unit_test_name_;             \
        int total_tests = 0;                                            \
        int total_failed_tests = 0;                                     \
        std::cout << "Starting Unit Test " << total_unit_tests << ": "  \
                  << unit_test_name << "\n";                            

/**
 * Finishes a unit test on a small unit of functionality, usually a function call.
 * Displays the number of failed tests, if any.
 * `UNIT_TEST` should be called before this.
 */
#define END_UNIT_TEST                                                   \
        if( total_failed_tests > 0 ) {                                  \
            total_failed_unit_tests++;                                  \
            std::cerr << TFS                                            \
                      << total_failed_tests << " out of "               \
                      << total_tests << " tests failed"                 \
                      << TFE << "\n";                                   \
        }                                                               \
    }


#define test_equals( _A, _B ) \
    total_tests++; \
    try { \
        const auto A = _A; \
        const auto B = _B; \
        if( A != B ) { \
            std::cerr << TFS << "Test Equals failed at line " << __LINE__ << "; A=" << std::to_string(A) << " B=" << std::to_string(B) << TFE << "\n"; \
            total_failed_tests++; \
        } \
    } catch(...) { \
        std::cerr << TFS << "Test Equals threw exception at line " << __LINE__ << TFE << "\n"; \
        total_failed_tests++; \
    }

#define test_float_equals( _A, _B ) \
    total_tests++; \
    try { \
        const float A = _A; \
        const float B = _B; \
        if( A - 1e-6 > B || B > A + 1e-6 ) { \
            std::cerr << TFS << "Test Float Equals failed at line " << __LINE__ << "; A=" << A << " B=" << B << TFE << "\n"; \
            total_failed_tests++; \
        } \
    } catch(...) { \
        std::cerr << TFS << "Test Equals threw exception at line " << __LINE__ << TFE << "\n"; \
        total_failed_tests++; \
    }

#define test_true( _Expression ) \
    total_tests++; \
    try { \
        if( !(_Expression) ) { \
            std::cerr << TFS << "Test True failed at line " << __LINE__ << TFE << "\n"; \
            total_failed_tests++; \
        } \
    } catch(...) { \
        std::cerr << "Test True threw exception at line " << __LINE__ << TFE << "\n"; \
        total_failed_tests++; \
    }

#define test_not_throws( _Code_Block ) \
    total_tests++; \
    try { \
        _Code_Block; \
    } catch(...) { \
        std::cerr << TFS << "Assert Not Throws failed at line " << __LINE__ << TFE << "\n"; \
        total_failed_tests++; \
    }

#define test_throws( _Code_Block ) \
    total_tests++; \
    try { \
        _Code_Block; std::cerr << TFS << "Assert Throws failed at line " << __LINE__ << TFE << "\n"; \
        total_failed_tests++; \
    } catch(...) { }
