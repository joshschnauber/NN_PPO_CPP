/* Tensor.hpp */

/**
 *  INFO:
 *  - A tensor with a rank RANK can be thought of as a vector where each element is a
 *    tensor of rank RANK-1, where a tensor of rank 1 is just a regular vector.
 *  - Most of the time, Tensor and VTensor are what will be used.
 *    Tensor and VTensor both inherit from BaseTensor, and they represent strictly 
 *    rectangular tensors.
 *  - A Tensor is a standard rectangular tensor, that owns and manages its own data.
 *  - A VTensor is a view tensor, essentially a reference to a part or whole of another 
 *    BaseTensor. A VTensor can reference another VTensor, but they are ultimately backed
 *    by a Tensor. VTensors themselves are always rectangular tensors.
 *  - A RaggedTensor is a Tensor where each inner tensor can be different sizes. Each of
 *    those inner Tensor are themselves rectangular though.
 */ 

/**
 * TODO:
 * - Some way to construct a tensor from a variable number of inner tensors, without the
 *   initializer list
 * - Some way to 'append' Tensors of the same or smaller dimension together 
 *   (i.e. inserting a row to the beginning of a matrix)
 * 
 * - Add Vector comparison with scalars, which returns Vector with 0 for false and 1 for 
 *   true.
 *     .operator > ()  and  .operator < ()  and  .operator == ()  and .operator != ()
 * - Add Vector indexing, to return a Tensor with only the selected 'rows' (select 
 *   'rows' in which the Vector has value > 0). Ideally this would return a VTensor.
 *     .select(Tensor, threshold = 0)
 * 
 * - Add ability to select multiple 'rows' from their index and return a Tensor
 *   containing only those rows.
 * - Add ability to select two indexes and return a Tensor of the rectangle selected.
 *     .operator [] (start[RANK], end[RANK])  or  .operator [] (start, end)
 * 
 * - Improve VTensor to allow representation of more complex shapes inside Tensors 
 *   (would require large refactor, especially of BaseTensor).
 *   Maybe switch BaseTensor& to generic TENSOR_TYPE in function calls to allow more 
 *   efficiently having different implementions for getting data, rather than virtual
 *   functions
 *   Ideally any 'shape changing' operation that doesn't change the values in the
 *   Tensor should return a VTensor (i.e. .transpose()), but this may not be feasible.
 * - Add constant version of VTensor
 * 
 * - Somehow 'connect' Tensor and VTensor with RaggedTensor, so their overlapping 
 *   functionality can be reused, and to get rid of the need for duplicated code.
 * 
 * - Add .collapsed() function that collapses Tensors with rank RANK into Tensor with rank
 *   RANK-1, when RANK>1.
 * - Add a function to fill the Tensor with random values
 * 
 * - Add .copy() functions that returns a new Tensor, and make more non size changing
 *   mutators alter the Tensor instead of creating a new one.
 * 
 * - Remove constructors that have fill param, and replace them with the .fill() function.
 *   The .fill() function needs to be able to return a reference to the type of object
 *   it is being called on however, and not always just a BaseTensor.  
 * - Maybe add fill method that accepts a pointer as well.
 */



#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <initializer_list>
#include <vector>
#include <cassert>

#include <type_traits>
#include <concepts>
#include <utility>



#ifndef TENSOR_HPP
#define TENSOR_HPP



/* Declaration */
namespace jai {
    /**
     * Tensor Declarations 
     */
    template<size_t RANK, typename NUM_T = float>
    class BaseTensor;
    template<size_t RANK, typename NUM_T = float>
    class Tensor;
    template<size_t RANK, typename NUM_T = float>
    class VTensor;
    template<size_t RANK, typename NUM_T = float>
    class RaggedTensor;


    /** 
     * Vector and Matrix type definitions.
     * All values store `float`s.
     */
    using BaseVector = BaseTensor<1, float>;
    using Vector = Tensor<1, float>;
    using VVector = VTensor<1, float>;
    using BaseMatrix = BaseTensor<2, float>;
    using Matrix = Tensor<2, float>;
    using VMatrix = VTensor<2, float>;
    using RaggedMatrix = RaggedTensor<2, float>;


    /**
     * Helper structs to define a type to represent the inner element of a Tensor.
     */
    namespace {
        template <size_t RANK, typename NUM_T>
        struct InnerElementType {
            using type = VTensor<RANK-1, NUM_T>;
        };
        template <typename NUM_T>
        struct InnerElementType<1, NUM_T> {
            using type = NUM_T&;
        };

        template <size_t RANK, typename NUM_T>
        struct ConstInnerElementType {
            using type = const VTensor<RANK-1, NUM_T>;
        };
        template <typename NUM_T>
        struct ConstInnerElementType<1, NUM_T> {
            using type = const NUM_T&;
        };
    }
    /**
     * Type used to represent the inner element of a Tensor of rank `RANK`.
     * An `InnerElement<RANK, NUM_T>` is a `VTensor<RANK-1, NUM_T>` if RANK>1,
     * and is a `NUM_T&` if RANK=1.
     */
    template <size_t RANK, typename NUM_T>
    using InnerElement = typename InnerElementType<RANK, NUM_T>::type;
    /**
     * Type used to represent the inner element of a constant Tensor of rank `RANK`.
     * An `InnerElement<RANK, NUM_T>` is a `const VTensor<RANK-1, NUM_T>` if RANK>1,
     * and is a `const NUM_T&` if RANK=1.
     */
    template <size_t RANK, typename NUM_T>
    using ConstInnerElement = typename ConstInnerElementType<RANK, NUM_T>::type;


    /**
     * Base class for Tensor iterators
     */
    namespace {
        template<typename Tensor_t>
        class BaseIterator {
            /* Constructors */
            protected:

            BaseIterator( const Tensor_t& tensor, size_t index = 0 );

            /* Public Functions */
            public:

            BaseIterator<Tensor_t>& operator ++ ();
            BaseIterator<Tensor_t> operator ++ (int);
            BaseIterator<Tensor_t>& operator -- ();
            BaseIterator<Tensor_t> operator -- (int);

            bool operator == ( const BaseIterator<Tensor_t>& other ) const;
            bool operator != ( const BaseIterator<Tensor_t>& other ) const;


            /* Member Variables */
            protected:

            Tensor_t* tensor;
            size_t index;
        };
    }
    /**
     * Tensor iterator.
     * This class can be used for any type of Tensor, even RaggedTensors.
     */
    template<typename Tensor_t>
    class TensorTypeIterator : public BaseIterator<Tensor_t> {
        /* Constructors */
        public:

        TensorTypeIterator( const Tensor_t& tensor, size_t index = 0 );
        TensorTypeIterator( const TensorTypeIterator<Tensor_t>& other );

        /* Public Functions */
        public:

        InnerElement<Tensor_t::Rank, typename Tensor_t::Num_t> operator * () const;
    };
    /**
     * Constant tensor iterator.
     * This class can be used for any type of Tensor, even RaggedTensors.
     */
    template<typename Tensor_t>
    class ConstTensorTypeIterator : public BaseIterator<Tensor_t> {
        /* Constructors */
        public: 

        ConstTensorTypeIterator( const Tensor_t& tensor, size_t index = 0 );
        ConstTensorTypeIterator( const BaseIterator<Tensor_t>& other );

        /* Public Functions */
        public:

        ConstInnerElement<Tensor_t::Rank, typename Tensor_t::Num_t> operator * () const;
    };


    /**
     * Helper structs to recursively define a type for initializing a Tensor with elements.
     */
    namespace {
        template<size_t RANK, typename NUM_T>
        struct InitializerElementsType {
            using type = std::initializer_list<typename InitializerElementsType<RANK-1, NUM_T>::type>;
        };
        template <typename NUM_T>
        struct InitializerElementsType<1, NUM_T> {
            using type = std::initializer_list<NUM_T>;
        };
    }
    /**
     * Type used to initialize a Tensor of rank `RANK` with elements.
     * An `InitializerElements<RANK, NUM_T>` contains a set of `InitializerElements<RANK-1, NUM_T>`s,
     * and an `InitializerElements<1, NUM_T>` contains a set of `NUM_T`s.
     */
    template <size_t RANK, typename NUM_T>
    using InitializerElements = typename InitializerElementsType<RANK, NUM_T>::type;


    /**
     * This defines the interface with a Tensor.
     * This is an abstract class that cannot be constructed on it's own.
     */
    template<size_t RANK, typename NUM_T>
    class BaseTensor {
        // Ensure that Tensor RANK cannot be 0 (must have 1 or more dimensions)
        static_assert(RANK > 0, "Tensor rank cannot be 0.");

        /* Member Types */
        public:

        static const size_t Rank = RANK;
        using Num_t = NUM_T;

        /* Constructors */
        protected:

        BaseTensor();

        /* Accessors */
        public:

        /**
         * Defined for RANK=1 Tensors, this returns the element at the given index in the
         * first (and only) dimension.
         */
        const NUM_T& operator [] ( size_t index ) const 
        requires (RANK == 1);
        /**
         * Defined for RANK=1 Tensors, this returns a mutable reference to the element at
         * the `index` in the first (and only) dimension.
         */
        NUM_T& operator [] ( size_t index )
        requires (RANK == 1);
        /**
         * Defined for RANK>1 Tensors, returns the element at the given indexes.
         */
        const NUM_T& operator [] ( const size_t (&indexes)[RANK] ) const;
        /**
         * Defined for RANK>1 Tensors, this returns a mutable reference to the element
         * at the given `indexes`.
         */
        NUM_T& operator [] ( const size_t (&indexes)[RANK] );
        /**
         * Defined for RANK>1 Tensors, this returns an immutable View Tensor with rank
         * RANK-1, at the given index in the first dimension.
         * The returned View Tensor is backed by `this` Tensor.
         */
        const VTensor<RANK-1, NUM_T> operator [] ( size_t index ) const
        requires (RANK > 1);
        /**
         * Defined for RANK>1 Tensors, this returns a View Tensor with rank RANK-1, at
         * the given index in the first dimension.
         * The returned View Tensor is backed by `this` Tensor.
         */
        VTensor<RANK-1, NUM_T> operator [] ( size_t index )
        requires (RANK > 1);
        /**
         * This returns a View Tensor with rank RANK containing the elements in `this`
         * Tensor from index `index_A`, inclusive, to index `index_B`, exclusive.
         * `index_B` must be greater than `index_A`.
         */
        const VTensor<RANK, NUM_T> slice( size_t index_A, size_t index_B ) const;

        /**
         * This returns a constant iterator corresponding to the first element of the first
         * dimension of `this` Tensor.
         */
        ConstTensorTypeIterator<BaseTensor<RANK, NUM_T>> begin() const;
        /**
         * This returns a constant iterator corresponding to one past the last element of
         * the first dimension of `this` Tensor.
         */
        ConstTensorTypeIterator<BaseTensor<RANK, NUM_T>> end() const;
        /**
         * This returns a iterator corresponding to the first element of the first
         * dimension of `this` Tensor.
         */
        TensorTypeIterator<BaseTensor<RANK, NUM_T>> begin();
        /**
         * This returns a iterator corresponding to one past the last element of
         * the first dimension of `this` Tensor.
         */
        TensorTypeIterator<BaseTensor<RANK, NUM_T>> end();

        /**
         * Returns an immutable View Tensor which is backed by `this` Tensor.
         */
        const VTensor<RANK, NUM_T> view() const;
        /**
         * Returns a View Tensor which is backed by `this` Tensor.
         */
        VTensor<RANK, NUM_T> view();
        /**
         * Returns an immutable View Tensor with rank RANK+1 of `this` Tensor, where it's
         * last dimension is of size 1.
         * The returned View Tensor is backed by `this` Tensor.
         * Useful for converting a Vector into an (n x 1) Matrix for matrix multiplication.
         */
        const VTensor<RANK+1> rankUp() const;
        /**
         * Returns a View Tensor with rank RANK+1 of `this` Tensor, where it's last
         * dimension is of size 1.
         * The returned View Tensor is backed by `this` Tensor.
         * Useful for converting a Vector into an (n x 1) Matrix for matrix multiplication.
         */
        VTensor<RANK+1> rankUp();
        /**
         * Returns an immutable View Tensor with rank 1 of `this` Tensor, all of it's
         * values are flattened into one vector.
         * The returned View Tensor is backed by `this` Tensor.
         */
        const VTensor<1, NUM_T> flattened() const;
        /**
         * Returns a View Tensor with rank 1 of `this` Tensor, all of it's values are
         * flattened into one vector.
         * The returned View Tensor is backed by `this` Tensor.
         */
        VTensor<1, NUM_T> flattened();

        /* Binary Operations */
        public:

        /**
         * Adds all of the elements in the `other` Tensor to all of the elements in
         * `this` Tensor and returns the result.
         * The `other` Tensor must be the same size as `this` Tensor.
         * The dimensions of `this` Tensor are passed onto the result Tensor.
         */
        Tensor<RANK, NUM_T> operator + ( const BaseTensor<RANK, NUM_T>& other ) const;
        /**
         * Subtracts all of the elements in the `other` Tensor from all of the elements
         * in `this` Tensor and returns the result.
         * The `other` Tensor must be the same size as `this` Tensor.
         * The dimensions of `this` Tensor are passed onto the result Tensor.
         */
        Tensor<RANK, NUM_T> operator - ( const BaseTensor<RANK, NUM_T>& other ) const;
        /**
         * Multiplies all of the elements in the `other` Tensor with all of the elements
         * in `this` Tensor and returns the result.
         * The `other` Tensor must be the same size as `this` Tensor.
         * The dimensions of `this` Tensor are passed onto the result Tensor.
         */
        Tensor<RANK, NUM_T> operator * ( const BaseTensor<RANK, NUM_T>& other ) const;
        /**
         * Divides all of the elements in the `other` Tensor from all of the elements
         * in `this` Tensor and returns the result.
         * The `other` Tensor must be the same size as `this` Tensor.
         * The dimensions of `this` Tensor are passed onto the result Tensor.
         */
        Tensor<RANK, NUM_T> operator / ( const BaseTensor<RANK, NUM_T>& other ) const;
        /**
         * Multiplies all of the elements in `this` Tensor by `scale` and returns the result.
        */
        template<size_t R, typename N_t>
        friend Tensor<R, N_t> operator * ( const BaseTensor<R, N_t>& tensor, typename BaseTensor<R, N_t>::Num_t scale );
        /**
         * Multiplies all of the elements in `this` Tensor by `scale` and returns the result.
        */
        template<size_t R, typename N_t>
        friend Tensor<R, N_t> operator * ( typename BaseTensor<R, N_t>::Num_t scale, const BaseTensor<R, N_t>& tensor );
        /**
         * Divides all of the elements in `this` Tensor by `scale` and returns the result.
        */
        Tensor<RANK, NUM_T> operator / ( NUM_T scale ) const;
        /**
         * Negates all of the elements in `this` Tensor and returns the result.
         */
        Tensor<RANK, NUM_T> operator - () const;
        /**
         * Returns a copy of `this` Tensor with the same dimensions, but with no data set.
         */
        Tensor<RANK, NUM_T> emptied() const;

        /**
         * Returns true if the `other` Tensor has the same dimensions as `this` Tensor, and false otherwise.
         */
        bool isSameSize( const BaseTensor<RANK, NUM_T>& other ) const;
        /**
         * Returns true if the `other` Tensor is equal to `this` Tensor, and false otherwise.
         */
        bool operator == ( const BaseTensor<RANK, NUM_T>& other ) const;
        /**
         * Returns true if the `other` Tensor is not equal to `this` Tensor, and false otherwise.
         */
        bool operator != ( const BaseTensor<RANK, NUM_T>& other ) const;

        /* General mutators */
        public:

        /** 
         * Adds all of the elements in the other Tensor to all of the elements in `this`
         * Tensor.
         * The `other` Tensor must be the same size as `this` Tensor.
         * necessarily have to have the same dimensions.
         */
        BaseTensor<RANK, NUM_T>& operator += ( const BaseTensor<RANK, NUM_T>& other );
        /**
         * Subtracts all of the elements in the other Tensor from all of the elements in
         * `this` Tensor.
         * The `other` Tensor must be the same size as `this` Tensor. 
         * necessarily have to have the same dimensions.
         */
        BaseTensor<RANK, NUM_T>& operator -= ( const BaseTensor<RANK, NUM_T>& other );
        /** 
         * Multiplies all of the elements in the `other` Tensor with all of the elements
         * in `this` Tensor.
         * The `other` Tensor must be the same size as `this` Tensor.
         * necessarily have to have the same dimensions.
         */
        BaseTensor<RANK, NUM_T>& operator *= ( const BaseTensor<RANK, NUM_T>& other );
        /**
         * Divides all of the elements in the `other` Tensor from all of the elements in
         * `this` Tensor.
         * The `other` Tensor must be the same size as `this` Tensor. 
         * necessarily have to have the same dimensions.
         */
        BaseTensor<RANK, NUM_T>& operator /= ( const BaseTensor<RANK, NUM_T>& other );
        /** 
         * Multiples all of the elements in `this` Tensor with the given `scale`.
         */
        BaseTensor<RANK, NUM_T>& operator *= ( NUM_T scale );
        /** 
         * Divides all of the elements in `this` Tensor with the given `scale`.
         */
        BaseTensor<RANK, NUM_T>& operator /= ( NUM_T scale );
        /**
         * This sets every value in `this` Tensor to `fill`.
         */
        BaseTensor<RANK, NUM_T>& fill( const NUM_T fill );
        /**
         * This sets the values in `this` Tensor to the values in `other`.
         * The `other` Tensor must be the same size as `this` Tensor.
         */
        BaseTensor<RANK, NUM_T>& set( const BaseTensor<RANK, NUM_T>& other );
        /**
         * This transforms each element in `this` Tensor using the given 
         * `transform_function`, which should return a `NUM_T`.
         * `transform_function` can have an argument of type `NUM_T` and/or an argument 
         * for the index (a `size_t` when RANK=1, and `size[RANK] when RANK>1`). The
         * arguments can be one or the other, and in any order.
         * The value in the Tensor is set to the returned value of `transform_function`.
         */
        template<typename Func>
        BaseTensor<RANK, NUM_T>& transform( Func transform_function );

        /* Convienience Operations */

        /**
         * Finds the mean of the elements in this Tensor and returns the result.
         */
        NUM_T mean() const;

        /* Vector Operations */
        public:

        /**
         * Finds the magnitude of this Vector and returns the result.
         */
        NUM_T mag() const
        requires (RANK == 1);
        /**
         * Finds the squared magnitude of this Vector and returns the result.
         */
        NUM_T squaredMag() const
        requires (RANK == 1);
        /**
         * Normalizes `this` Vector, and returns the result.
         */
        Tensor<1, NUM_T> normalized() const
        requires (RANK == 1);
        /**
         * Takes the dot product of this Vector with the other Vector and returns the result.
         * The two vectors must be the same size.
         */
        NUM_T dot( const BaseTensor<1, NUM_T>& other ) const
        requires (RANK == 1);
        /**
         * Takes the cross product of this Vector with the other Vector and returns the result.
         * The two vectors must have a size of 3.
         */
        Tensor<1, NUM_T> cross( const BaseTensor<1, NUM_T>& other ) const
        requires (RANK == 1);

        /* Matrix Operations */
        public:

        /**
         * Takes the transpose of `this` Matrix and returns the result.
         * If `this` Matrix is of size (m x n), then the result will be of size (n x m).
         */
        Tensor<2, NUM_T> transpose() const
        requires (RANK == 2);
        /**
         * Takes the transpose of `this` Vector and returns the result.
         * If `this` Vector of is size (m), then the result will be of size (1 x m).
         */
        Tensor<2, NUM_T> transpose() const
        requires (RANK == 1);
        /**
         * Finds the matrix multiplication of the `other` Matrix on `this` Matrix and
         * returns the result.
         * `this` Matrix must be of size (m x n) and the `other` Matrix must be of size
         * (n x w)
         */
        Tensor<2, NUM_T> mul( const BaseTensor<2, NUM_T>& other ) const
        requires (RANK == 2);
        /**
         * Finds the matrix multiplication of the `other` Vector on `this` Matrix and
         * returns the result.
         * `this` matrix must be of size (m x n) and the `other` Vector must be of size
         * (n).
         */
        Tensor<1, NUM_T> mul( const BaseTensor<1, NUM_T>& other ) const
        requires (RANK == 2);
        /**
         * Finds the matrix multiplication of the `other` Matrix on `this` Vector and 
         * returns the result.
         * `this` vector must be of size (m) and the `other` matrix must be of size
         * (1 x n).
         */
        Tensor<2, NUM_T> mul( const BaseTensor<2, NUM_T>& other ) const
        requires (RANK == 1);
        /**
         * Finds the determinant of `this` Matrix and returns the result. 
         * `this` Matrix must be of size (n x n).
         */
        NUM_T determinant() const
        requires (RANK == 2);
        /**
         * Finds the matrix inverse of `this` Matrix and returns the result. 
         * `this` Matrix must be of size (n x n) and invertible (the columns are linearly
         * independent).
         */
        Tensor<2, NUM_T> inverse() const
        requires (RANK == 2);

        /* Getters */
        public:

        /**
         * Returns the rank of the tensor (the number of dimensions).
         * NOTE: This is NOT the same as the matrix rank.
         */
        constexpr size_t rank() const;
        /**
         * Returns the total size of the Tensor (the total number of elements).
         */
        size_t totalSize() const;
        /**
         * Returns a pointer to the start of the contiguous data stored in the Tensor.
         */
        const NUM_T* data() const;
        /**
         * This returns the size of first dimension of the Tensor.
         * For RANK=1 Tensors, this is the same as calling totalSize().
         */
        size_t size() const;
        /**
         * This returns the size of the given dimension.
         */
        size_t size( size_t dim_index ) const;

        /**
         * Prints out the Tensor as a string.
         */
        template<size_t R, typename N_t>
        friend std::ostream& operator << ( std::ostream& fs, const BaseTensor<R, N_t>& t );

        /* Member Variables */
        protected:
        
        /**
         * The total number of elements in the Tensor.
         * This is the size of each dimension multiplied together.
         */
        size_t total_size;
        /**
         * The pointer to the allocated data in this Tensor.
         * The memory from `data` to `data + total_size - 1` will always be valid.
         */
        NUM_T* data_;
        /**
         * The size of each dimension of the Tensor.
         */
        size_t dimensions[RANK];

        /* Friend Classes */
        public:
        
        /**
         * Declare friend classes so that base/derived classes can access each others
         * internal data.
         */
        template<size_t R, typename N_t>
        friend class BaseTensor;
        template<size_t R, typename N_t>
        friend class Tensor;
        template<size_t R, typename N_t>
        friend class VTensor;
        /**
         * Declare RaggedTensor as a friend of BaseTensor so that it can view and manage
         * internal Tensor and VTensor data.
         */
        template<size_t R, typename N_t>
        friend class RaggedTensor;
    };


    /**
     * This represents a Tensor itself, which contains and manages all of it's own data.
     * Any instance of a Tensor, and the data contained within, is managed by itself.
     */
    template<size_t RANK, typename NUM_T>
    class Tensor : public BaseTensor<RANK, NUM_T> {
        /* Constructors */
        public:

        /**
         * Constructs an empty Tensor with a size of 0 in each dimension.
         */
        Tensor();
        /**
         * Defined for RANK=1 Tensors, constructs a Tensor with the given dimension.
         * Throws an error if `dim` is equal to 0.
         */
        explicit 
        Tensor( size_t dim )
        requires (RANK == 1);
        /**
         * Defined for RANK=1 Tensors, constructs a Tensor with the given dimensions and
         * with all values set to `fill`.
         * Throws an error if `dim` is equal to 0.
         */
        Tensor( size_t dim, NUM_T fill )
        requires (RANK == 1);
        /**
         * Defined for RANK=1 Tensors, constructs a Tensor with the given dimensions and
         * set with the values from `fill`. `fill` must be have valid memory from index 0
         * to `dim-1`.
         * Throws an error if `dim` is equal to 0.
         */
        Tensor( size_t dim, const NUM_T fill[] )
        requires (RANK == 1);
        /**
         * Constructs a Tensor with the given dimensions.
         * Throws an error if any value in `dims` is equal to 0.
         */
        explicit 
        Tensor( const size_t (&dims)[RANK] );
        /**
         * Constructs a Tensor with the given dimensions and with all values set to
         * `fill`.
         * Throws an error if any value in `dims` is equal to 0.
         */
        Tensor( const size_t (&dims)[RANK], NUM_T fill );
        /**
         * Constructs a Tensor initialized with the given `elements`.
         * Throws an error if `elements` or any inner elements inside `elements` has a
         * size of 0.
         * Throws an error if the `elements` are non-rectangular.
         */
        Tensor( InitializerElements<RANK, NUM_T> elements );
        /**
         * Defined for RANK>1 Tensors, constructs a Tensor initialized with the given
         * `Tensor<RANK-1, NUM_T>` elements. The size of the first dimension is the size of
         * `elements`.
         * Throws an error if `elements` has a size of 0.
         * Throws an error if any of the Tensors in `elements` have differing dimensions.
         */
        Tensor( std::initializer_list<std::reference_wrapper<const BaseTensor<RANK-1, NUM_T>>> elements )
        requires (RANK > 1);
        /**
         * Defined for RANK=1 Tensors, constructs a Tensor initialized with the elements
         * in the `std::vector<NUM_T>`
         */
        Tensor( const std::vector<NUM_T>& vec )
        requires (RANK == 1);

        /**
         * Copy constructor from BaseTensor.
         */
        Tensor( const BaseTensor<RANK, NUM_T>& other );
        /**
         * Copy constructor.
         */
        Tensor( const Tensor<RANK, NUM_T>& other );
        /**
         * Move constructor.
         */
        Tensor( Tensor<RANK, NUM_T>&& other );
        /**
         * Destructor.
         */
        ~Tensor();
        /**
         * Assignment operator from BaseTensor.
         * Ensures that memory is freed when existing object is overwritten.
         * Any VTensors referring to `this` Tensor will be invalidated.
         */
        Tensor<RANK, NUM_T>& operator = ( const BaseTensor<RANK, NUM_T>& other );
        /**
         * Assignment operator.
         * Ensures that memory is freed when existing object is overwritten.
         * Any VTensors referring to `this` Tensor will be invalidated.
         */
        Tensor<RANK, NUM_T>& operator = ( const Tensor<RANK, NUM_T>& other );
        /**
         * Move assignment operator.
         * Ensures that memory is freed when existing object is overwritten.
         * Any VTensors referring to `this` Tensor will be invalidated.
         */
        Tensor<RANK, NUM_T>& operator = ( Tensor<RANK, NUM_T>&& other );
    
        /* Factory functions */
        public:

        /**
         * Creates a Vector of size `dim` with values evenly spaced between `min` and 
         * `max`, such that `min` is the first element and `max` is the last.
         * `min` must be less than or equal to `max`, and `step_size` must be greater 
         * than 0.
         */
        static Tensor<1, NUM_T> range( size_t dim, NUM_T min, NUM_T max );
        /**
         * Creates a square identity (`dims` x `dims`) matrix with the given
         * `diagonal_value`.
         */
        static Tensor<2, NUM_T> identity( size_t dims, NUM_T diagonal_value = 1.0f );
    };


    /**
     * This represents the view into a part or whole of a Tensor.
     * Any instance of a VTensor, and the data contained within, is backed by a Tensor.
     * Despite its name, a VTensor can be modified, but it will also modify the Tensor it is backed by.
     */
    template<size_t RANK, typename NUM_T>
    class VTensor : public BaseTensor<RANK, NUM_T> {
        /* Constructors */
        public:

        /**
         * Constructs an empty VTensor with a size of 0 in each dimension.
         */
        VTensor();
        /**
         * Copy constructor.
         */
        VTensor( const VTensor<RANK, NUM_T>& other );
        /**
         * Assignment operator.
         */
        VTensor<RANK, NUM_T>& operator = ( const VTensor<RANK, NUM_T>& other );
    };


    /**
     * This represents a tensor whose first dimensions elements do not have the same
     * size. Each inner tensor has a rank of RANK-1, but can have differing sizes.
     * NOTE: Name stolen from PyTorch, though unsure if functionally similar.
     */
    template<size_t RANK, typename NUM_T>
    class RaggedTensor {
        // Ensure that Ragged Tensor RANK cannot be less than 1 (must have 2 or more dimensions)
        static_assert(RANK > 1, "Ragged Tensor rank cannot be less than 1.");

        /* Member Types */
        public:

        static const size_t Rank = RANK;
        using Num_t = NUM_T;

        /* Constructors */
        public:

        /**
         * Constructs an empty RaggedTensor with a size of 0 in each dimension.
         */
        RaggedTensor();
        /**
         * Defined for RANK=2 RaggedTensors, constructs a RaggedTensor containing inner
         * Tensors with the dimensions specified in `inner_tensor_dims`.
         */
        RaggedTensor( const size_t dim1_size, const size_t inner_tensor_dims[] )
        requires (RANK == 2);
        /**
         * Constructs a RaggedTensor containing inner Tensors with the set of dimensions
         * specified in `inner_tensor_dims`.
         */
        RaggedTensor( const size_t dim1_size, const size_t inner_tensor_dims[][RANK-1] );
        /**
         * Defined for RANK=2 RaggedTensors, constructs a RaggedTensor containing inner
         * Tensors with the dimensions specified in `inner_tensor_dims`.
         * Identical to the constructor using pointers, but does not require a separate
         * dimension 1 size field.
         */
        explicit 
        RaggedTensor( std::initializer_list<size_t> inner_tensor_dims )
        requires (RANK == 2);
        /**
         * Constructs a RaggedTensor containing inner Tensors with the set of dimensions
         * specified in `inner_tensor_dims`.
         * Identical to the constructor using pointers, but does not require a separate
         * dimension 1 size field.
         */
        explicit 
        RaggedTensor( std::initializer_list<size_t[RANK-1]> inner_tensor_dims );
        /**
         * Constructs a RaggedTensor initialized with the given `elements`.
         * Throws an error if `elements` or any inner elements inside `elements` has a
         * size of 0.
         * Throws an error if any of the elements inside `elements` are non-rectangular.
         */
        RaggedTensor( InitializerElements<RANK, NUM_T> elements );
        /**
         * Constructs a RaggedTensor containing the tensors specified in `elements`.
         * Throws an error if `elements` has a size of 0.
         */
        RaggedTensor( std::initializer_list<std::reference_wrapper<const BaseTensor<RANK-1, NUM_T>>> elements );

        /**
         * Copy constructor.
         */
        RaggedTensor( const RaggedTensor<RANK, NUM_T>& other );
        /**
         * Copy constructor from BaseTensor.
         */
        RaggedTensor( const BaseTensor<RANK, NUM_T>& other );
        /**
         * Move constructor.
         */
        RaggedTensor( RaggedTensor<RANK, NUM_T>&& other );
        /**
         * Destructor.
         */
        ~RaggedTensor();
        /**
         * Assignment operator.
         * Ensures that memory is freed when existing object is overwritten.
         */
        RaggedTensor<RANK, NUM_T>& operator = ( const RaggedTensor<RANK, NUM_T>& other );
        /**
         * Assignment operator from BaseTensor.
         * Ensures that memory is freed when existing object is overwritten.
         */
        RaggedTensor<RANK, NUM_T>& operator = ( const BaseTensor<RANK, NUM_T>& other );
        /**
         * Move assignment operator.
         * Ensures that memory is freed when existing object is overwritten.
         */
        RaggedTensor<RANK, NUM_T>& operator = ( RaggedTensor<RANK, NUM_T>&& other );

        /* Accessors */
        public:

        /**
         * Returns the element at the given `indexes`.
         */
        const NUM_T& operator [] ( const size_t (&indexes)[RANK] ) const;
        /**
         * Returns a mutable reference to the element at the given `indexes`.
         */
        NUM_T& operator [] ( const size_t (&indexes)[RANK] );
        /**
         * This returns the inner Tensor at the given index in the first dimension.
         */
        const VTensor<RANK-1, NUM_T> operator [] ( size_t index ) const;
        /**
         * This returns a mutable reference to the inner Tensor at the given index in the
         * first dimension.
         */
        VTensor<RANK-1, NUM_T> operator [] ( size_t index );

        /**
         * This returns a constant iterator corresponding to the first element of the first
         * dimension of `this` RaggedTensor.
         */
        ConstTensorTypeIterator<RaggedTensor<RANK, NUM_T>> begin() const;
        /**
         * This returns a constant iterator corresponding to one past the last element of
         * the first dimension of `this` RaggedTensor.
         */
        ConstTensorTypeIterator<RaggedTensor<RANK, NUM_T>> end() const;
        /**
         * This returns a iterator corresponding to the first element of the first
         * dimension of `this` RaggedTensor.
         */
        TensorTypeIterator<RaggedTensor<RANK, NUM_T>> begin();
        /**
         * This returns a iterator corresponding to one past the last element of
         * the first dimension of `this` RaggedTensor.
         */
        TensorTypeIterator<RaggedTensor<RANK, NUM_T>> end();

        /* Binary Operations */
        public:

        /**
         * Adds all of the elements in the `other` RaggedTensor to all of the elements in
         * `this` RaggedTensor and returns the result.
         * The `other` RaggedTensor must be the same size as `this` RaggedTensor. 
         * The dimensions of `this` RaggedTensor are passed onto the result RaggedTensor.
         */
        RaggedTensor<RANK, NUM_T> operator + ( const RaggedTensor<RANK, NUM_T>& other ) const;
        /**
         * Subtracts all of the elements in the `other` RaggedTensor from all of the
         * elements in `this` RaggedTensor and returns the result.
         * The `other` RaggedTensor must be the same size as `this` RaggedTensor. 
         * The dimensions of `this` RaggedTensor are passed onto the result RaggedTensor.
         */
        RaggedTensor<RANK, NUM_T> operator - ( const RaggedTensor<RANK, NUM_T>& other ) const;
        /**
         * Multiplies all of the elements in the `other` RaggedTensor with all of the
         * elements in `this` RaggedTensor and returns the result.
         * The `other` RaggedTensor must be the same size as `this` RaggedTensor. 
         * The dimensions of `this` RaggedTensor are passed onto the result RaggedTensor.
         */
        RaggedTensor<RANK, NUM_T> operator * ( const RaggedTensor<RANK, NUM_T>& other ) const;
        /**
         * Divides all of the elements in the `other` RaggedTensor from all of the
         * elements in `this` RaggedTensor and returns the result.
         * The `other` RaggedTensor must be the same size as `this` RaggedTensor. 
         * The dimensions of `this` RaggedTensor are passed onto the result RaggedTensor.
         */
        RaggedTensor<RANK, NUM_T> operator / ( const RaggedTensor<RANK, NUM_T>& other ) const;
        /**
         * Multiplies all of the elements in `this` RaggedTensor by `scale` and returns
         * the result.
        */
        template<size_t R, typename N_t>
        friend RaggedTensor<R, N_t> operator * ( const RaggedTensor<R, N_t>& tensor, typename RaggedTensor<R, N_t>::Num_t scale );
        /**
         * Multiplies all of the elements in `this` RaggedTensor by `scale` and returns
         * the result.
        */
        template<size_t R, typename N_t>
        friend RaggedTensor<R, N_t> operator * ( typename RaggedTensor<R, N_t>::Num_t scale, const RaggedTensor<R, N_t>& tensor );
        /**
         * Divides all of the elements in `this` RaggedTensor by `scale` and returns the
         * result.
        */
        RaggedTensor<RANK, NUM_T> operator / ( NUM_T scale ) const;
        /**
         * Negates all of the elements in `this` RaggedTensor and returns the result.
         */
        RaggedTensor<RANK, NUM_T> operator - () const;
        /**
         * Returns a copy of `this` RaggedTensor with the same dimensions, but with no
         * data set.
         */
        RaggedTensor<RANK, NUM_T> emptied() const;

        /**
         * Returns true if the `other` RaggedTensor has the same dimensions as `this` RaggedTensor, and false otherwise.
         */
        bool isSameSize( const RaggedTensor<RANK, NUM_T>& other ) const;
        /**
         * Returns true if the `other` RaggedTensor is equal to `this` RaggedTensor, and false otherwise.
         */
        bool operator == ( const RaggedTensor<RANK, NUM_T>& other ) const;
        /**
         * Returns true if the `other` RaggedTensor is not equal to `this` RaggedTensor, and false otherwise.
         */
        bool operator != ( const RaggedTensor<RANK, NUM_T>& other ) const;
        
        /* General mutators */
        public:

        /** 
         * Adds all of the elements in the other RaggedTensor to all of the elements in
         * `this` RaggedTensor.
         * The `other` RaggedTensor must be the same size as `this` RaggedTensor.
         */
        RaggedTensor<RANK, NUM_T>& operator += ( const RaggedTensor<RANK, NUM_T>& other );
        /**
         * Subtracts all of the elements in the other RaggedTensor from all of the
         * elements in `this` RaggedTensor.
         * The `other` RaggedTensor must be the same size as `this` RaggedTensor.
         */
        RaggedTensor<RANK, NUM_T>& operator -= ( const RaggedTensor<RANK, NUM_T>& other );
        /** 
         * Multiplies all of the elements in the `other` RaggedTensor to all of the
         * elements in `this` RaggedTensor.
         * The `other` RaggedTensor must be the same size as `this` RaggedTensor.
         */
        RaggedTensor<RANK, NUM_T>& operator *= ( const RaggedTensor<RANK, NUM_T>& other );
        /**
         * Divides all of the elements in the `other` RaggedTensor from all of the
         * elements in `this` RaggedTensor.
         * The `other` RaggedTensor must be the same size as `this` RaggedTensor.
         */
        RaggedTensor<RANK, NUM_T>& operator /= ( const RaggedTensor<RANK, NUM_T>& other );
        /** 
         * Multiples all of the elements in `this` RaggedTensor with the given `scale`.
         */
        RaggedTensor<RANK, NUM_T>& operator *= ( NUM_T scale );
        /** 
         * Divides all of the elements in `this` RaggedTensor by the given `scale`.
         */
        RaggedTensor<RANK, NUM_T>& operator /= ( NUM_T scale );
        /**
         * This sets every value in `this` RaggedTensor to `fill`.
         */
        RaggedTensor<RANK, NUM_T>& fill( const NUM_T fill );
        /**
         * This sets the values in `this` RaggedTensor to the values in `other`.
         * The `other` RaggedTensor must be the same size as `this` RaggedTensor. 
         */
        RaggedTensor<RANK, NUM_T>& set( const RaggedTensor<RANK, NUM_T>& other );
        /**
         * This transforms each element in `this` RaggedTensor using the given 
         * `transform_function`. The only argument the function should take is of type
         * `NUM_T`, and the function should return a `NUM_T`.
         * The value in the RaggedTensor is set to the returned value of
         * `transform_function`.
         */
        template<typename Func>
        RaggedTensor<RANK, NUM_T>& transform( Func transform_function );

        /* Getters */
        public:

        /**
         * Returns the rank of the tensor (the number of dimensions).
         * NOTE: This is NOT the same as the matrix rank.
         */
        constexpr size_t rank() const;
        /**
         * Returns the total size of the RaggedTensor (the total number of elements).
         */
        size_t totalSize() const;
        /**
         * The pointer to the allocated data in this RaggedTensor.
         */
        const NUM_T* data() const;
        /**
         * Returns the size of the first dimension of this RaggedTensor.
         */
        size_t size() const;

        /**
         * Prints out the RaggedTensor as a string.
         */
        template<size_t R, typename N_t>
        friend std::ostream& operator << ( std::ostream& fs, const RaggedTensor<R, N_t>& rt );

        /* Member Variables */
        public:

        /**
         * The total number of elements in the RaggedTensor.
         * This is the sum of the total size of each inner Tensor.
         */
        size_t total_size;
        /**
         * The pointer to the allocated data in this Tensor.
         * The memory from `data` to `data + total_size - 1` will always be valid.
         */
        NUM_T* data_;
        /**
         * The size of the first dimension. More simply, the number of inner tensors
         */
        size_t dimension1;
        /**
         * The View Tensors which keep track of the size and memory locations of the
         * inner tensors.
         */
        VTensor<RANK-1, NUM_T>* inner_tensors;
    };
}



/* Implementation */
namespace jai {

    /* Iterator Implementation */

    template<typename Tensor_t>
    BaseIterator<Tensor_t>::BaseIterator( const Tensor_t& tensor, size_t index ) : 
        tensor(const_cast<Tensor_t*>(&tensor)), 
        index(index) 
    { }

    template<typename Tensor_t>
    BaseIterator<Tensor_t>& BaseIterator<Tensor_t>::operator ++ () {
        ++this->index;
        return *this;
    }
    
    template<typename Tensor_t>
    BaseIterator<Tensor_t> BaseIterator<Tensor_t>::operator ++ (int) {
        BaseIterator temp = *this;
        ++(*this);
        return temp;
    }

    template<typename Tensor_t>
    BaseIterator<Tensor_t>& BaseIterator<Tensor_t>::operator -- () {
        --this->index;
        return *this;
    }
    
    template<typename Tensor_t>
    BaseIterator<Tensor_t> BaseIterator<Tensor_t>::operator -- (int) {
        BaseIterator temp = *this;
        --(*this);
        return temp;
    }

    template<typename Tensor_t>
    bool BaseIterator<Tensor_t>::operator == ( const BaseIterator<Tensor_t>& other ) const { 
        return this->tensor == other.tensor  &&  this->index == other.index; 
    }

    template<typename Tensor_t>
    bool BaseIterator<Tensor_t>::operator != ( const BaseIterator<Tensor_t>& other ) const { 
        return !(*this == other); 
    }

    template<typename Tensor_t>
    TensorTypeIterator<Tensor_t>::TensorTypeIterator( const Tensor_t& tensor, size_t index ) : 
        BaseIterator<Tensor_t>(tensor, index) 
    { }

    template<typename Tensor_t>
    TensorTypeIterator<Tensor_t>::TensorTypeIterator( const TensorTypeIterator<Tensor_t>& other ) :
        BaseIterator<Tensor_t>(other.tensor, other.index)
    { }

    template<typename Tensor_t>
    InnerElement<Tensor_t::Rank, typename Tensor_t::Num_t> TensorTypeIterator<Tensor_t>::operator * () const { 
        Tensor_t* non_const_ptr = const_cast<Tensor_t*>(this->tensor);
        return (*non_const_ptr)[this->index];
    }

    template<typename Tensor_t>
    ConstTensorTypeIterator<Tensor_t>::ConstTensorTypeIterator( const Tensor_t& tensor, size_t index ) : 
        BaseIterator<Tensor_t>(tensor, index) 
    { }

    template<typename Tensor_t>
    ConstTensorTypeIterator<Tensor_t>::ConstTensorTypeIterator( const BaseIterator<Tensor_t>& other ) :
        BaseIterator<Tensor_t>(other.tensor, other.index) 
    { }

    template<typename Tensor_t>
    ConstInnerElement<Tensor_t::Rank, typename Tensor_t::Num_t> ConstTensorTypeIterator<Tensor_t>::operator * () const {
        const Tensor_t* const_ptr = const_cast<const Tensor_t*>(this->tensor);
        return (*const_ptr)[this->index]; 
    }


    /* Implementation Helper Functions for Handling Array References */
    
    namespace {
        template <size_t N>
        constexpr const size_t (&tail( const size_t (&array)[N] ))[N - 1] {
            return *reinterpret_cast<const size_t (*)[N - 1]>(&array[1]);
        }
    }


    /* Error Checking */

    namespace {
        /**
         * Checks that `index` is within the bounds of the first dimension of `tensor`.
         */
        template<typename Tensor_t>
        inline void debugCheckBound( 
            [[maybe_unused]] const Tensor_t& tensor, 
            [[maybe_unused]] const size_t index 
        ) {
        #ifdef DEBUG
            const size_t size = tensor.size();
            if( index >= size ) {
                throw std::out_of_range(
                    "Index " + std::to_string(index) + 
                    " is out of bounds for tensor with size " + std::to_string(size)
                );
            }
        #endif
        }

        /**
         * Checks that `indexes` is within the bounds of `tensor`.
         */
        template<typename Tensor_t>
        inline void debugCheckBound( 
            [[maybe_unused]] const Tensor_t& tensor, 
            [[maybe_unused]] const size_t (&indexes)[Tensor_t::Rank],
            [[maybe_unused]] const size_t dim_index = 0
        ) {
        #ifdef DEBUG
            const size_t index = indexes[0];
            const size_t size = tensor.size();
        
            if( index >= size ) {
                throw std::out_of_range(
                    "Index " + std::to_string(index) + 
                    " is out of bounds in dimension " + std::to_string(dim_index) +
                    " with size " + std::to_string(size)
                );
            }
            
            if constexpr( Tensor_t::Rank > 1 ) {
                debugCheckBound(tensor[index], tail(indexes), dim_index + 1);
            }
        #endif
        }

        /**
         * Checks that `dim_index` is within the bounds of the dimension sizes array.
         */
        template<typename Tensor_t>
        inline void debugCheckDimensionsBound( 
            [[maybe_unused]] const Tensor_t& tensor, 
            [[maybe_unused]] const size_t dim_index 
        ) {
        #ifdef DEBUG
            const size_t num_dims = tensor.rank();
            if( dim_index >= num_dims ) {
                throw std::out_of_range(
                    "Dimension index " + std::to_string(dim_index) + 
                    " is out of bounds for tensor with " + std::to_string(num_dims) +
                    " dimensions"
                );
            }
        #endif
        }

        /**
         * Checks that `tensor_A` and `tensor_B` are the same size.
         */
        template<typename Tensor_t>
        inline void debugCheckSizes(
            [[maybe_unused]] const Tensor_t& tensor_A,
            [[maybe_unused]] const Tensor_t& tensor_B
        ) {
        #ifdef DEBUG
            if( !tensor_A.isSameSize(tensor_B) ) {
                throw std::invalid_argument(
                    "The tensor's sizes do not match"
                );
            }
        #endif
        }

        /**
         * Checks that `tensor_A` and `tensor_B` can have a cross product performed
         * between them.
         */
        template<typename Tensor_t>
        inline void debugCheckCrossProduct(
            [[maybe_unused]] const Tensor_t& tensor_A,
            [[maybe_unused]] const Tensor_t& tensor_B
        ) {
        #ifdef DEBUG
            debugCheckSizes(tensor_A, tensor_B);
            if( tensor_A.rank() != 1 || tensor_B.rank() != 1 ) {
                throw std::invalid_argument(
                    "Cross product cannot be performed on non-vectors"
                );  
            }

            const size_t size_A = tensor_A.size();
            const size_t size_B = tensor_B.size();
            if( size_A != 3  ||  size_B != 3 ) {
                throw std::invalid_argument(
                    "Cross product cannot be performed between vectors of length " +
                    std::to_string(size_A) + " and length " + std::to_string(size_B)
                );  
            }
        #endif
        }

        /**
         * Checks that `tensor_A` and `tensor_B` can have a matrix multiplication
         * performed between them.
         */
        template<typename Tensor_t1, typename Tensor_t2>
        inline void debugCheckMatrixMult(
            [[maybe_unused]] const Tensor_t1& tensor_A,
            [[maybe_unused]] const Tensor_t2& tensor_B
        ) {
        #ifdef DEBUG
            if( tensor_A.rank() > 2 || tensor_B.rank() > 2 ) {
                throw std::invalid_argument(
                    "Matrix multiplication cannot be performed on non-matrices"
                );  
            }

            size_t size_dim1_A = tensor_A.size(0);
            size_t size_dim2_A = (tensor_A.rank() == 2) ? tensor_A.size(1) : 1;
            size_t size_dim1_B =  tensor_B.size(0);
            size_t size_dim2_B = (tensor_B.rank() == 2) ? tensor_B.size(1) : 1;
            if( size_dim2_A != size_dim1_B ) {
                throw std::invalid_argument(
                    "Matrix multiplication cannot be performed between matrices"
                    " of size (" + std::to_string(size_dim1_A) + "x" + 
                    std::to_string(size_dim2_A) + ")"
                    " and size (" + std::to_string(size_dim1_B) + "x" + 
                    std::to_string(size_dim2_B) + ")"
                );  
            }
        #endif
        }

        /**
         * Checks a determinant is defined for `tensor`.
         */
        template<typename Tensor_t>
        inline void debugCheckDeterminant(
            [[maybe_unused]] const Tensor_t& tensor
        ) {
        #ifdef DEBUG
            if( tensor.rank() != 2 ) {
                throw std::invalid_argument(
                    "The determinant is not defined for non matrices"
                );  
            }

            size_t size_dim1 = tensor.size(0);
            size_t size_dim2 =  tensor.size(1);
            if( size_dim1 != size_dim2 ) {
                throw std::invalid_argument(
                    "The determinant is not defined for " + std::to_string(size_dim1) +
                    " by " + std::to_string(size_dim2) + " matrices"
                );  
            }
        #endif
        }
    }
    

    /* Implementation Helper Functions For Bulk Operations */

    namespace {
        template<typename NUM_T>
        inline void fillValues( const NUM_T A, NUM_T* dest, const size_t size ) {
            std::fill(dest, dest + size, A);
        }
        template<typename NUM_T>
        inline void setValues( const NUM_T* src_A, NUM_T* dest, const size_t size ) {
            std::memcpy( dest, src_A, size * sizeof(NUM_T) );
        }
        template<typename NUM_T>
        inline void addValues( const NUM_T* src_A, const NUM_T* src_B, NUM_T* dest, const size_t size ) {
            for( size_t i = 0; i < size; ++i ) {
                dest[i] = src_A[i] + src_B[i];
            }
        }
        template<typename NUM_T>
        inline void subtractValues( const NUM_T* src_A, const NUM_T* src_B, NUM_T* dest, const size_t size ) {
            for( size_t i = 0; i < size; ++i ) {
                dest[i] = src_A[i] - src_B[i];
            }
        }
        template<typename NUM_T>
        inline void multiplyValues( const NUM_T* src_A, const NUM_T* src_B, NUM_T* dest, const size_t size ) {
            for( size_t i = 0; i < size; ++i ) {
                dest[i] = src_A[i] * src_B[i];
            }
        }
        template<typename NUM_T>
        inline void divideValues( const NUM_T* src_A, const NUM_T* src_B, NUM_T* dest, const size_t size ) {
            for( size_t i = 0; i < size; ++i ) {
                dest[i] = src_A[i] / src_B[i];
            }
        }
        template<typename NUM_T>
        inline void multiplyValuesByScalar( const NUM_T* src_A, const NUM_T B, NUM_T* dest, const size_t size ) {
            for( size_t i = 0; i < size; ++i ) {
                dest[i] = src_A[i] * B;
            }
        }
        template<typename NUM_T>
        inline void negateValues( const NUM_T* src_A, NUM_T* dest, const size_t size ) {
            for( size_t i = 0; i < size; ++i ) {
                dest[i] = -src_A[i];
            }
        }
        template<typename NUM_T>
        inline bool compareValuesForEquality( const NUM_T* src_A, const NUM_T* src_B, const size_t size ) {
            for( size_t i = 0; i < size; ++i ) {
                if( src_A[i] != src_B[i] ) {
                    return false;
                }
            }
            return true;
        }
    }


    /* BaseTensor Implementation */

    template<size_t RANK, typename NUM_T>
    BaseTensor<RANK, NUM_T>::BaseTensor() { }

    template<size_t RANK, typename NUM_T>
    const NUM_T& BaseTensor<RANK, NUM_T>::operator [] ( const size_t index ) const 
    requires (RANK == 1) {
        debugCheckBound(*this, index);

        return this->data_[index];
    }
    
    template<size_t RANK, typename NUM_T>
    NUM_T& BaseTensor<RANK, NUM_T>::operator [] ( const size_t index ) 
    requires (RANK == 1) {
        debugCheckBound(*this, index);

        return this->data_[index];
    }
    
    template<size_t RANK, typename NUM_T>
    const NUM_T& BaseTensor<RANK, NUM_T>::operator [] ( const size_t (&indexes)[RANK] ) const {
        debugCheckBound(*this, indexes);

        size_t index = 0;
        size_t inner_tensor_size = 1;
        for( size_t i = 0; i < RANK; ++i ) {
            index += inner_tensor_size * indexes[RANK - i - 1];
            inner_tensor_size *= this->dimensions[RANK - i - 1];
        }
        return this->data_[index];
    }
    
    template<size_t RANK, typename NUM_T>
    NUM_T& BaseTensor<RANK, NUM_T>::operator [] ( const size_t (&indexes)[RANK] ) {
        // Get value from const version of this function
        const BaseTensor<RANK, NUM_T>* const_this = static_cast<const BaseTensor<RANK, NUM_T>*>(this);
        const NUM_T& const_val = const_this->operator[](indexes);
        return const_cast<NUM_T&>(const_val);
    }
    
    template<size_t RANK, typename NUM_T>
    const VTensor<RANK-1, NUM_T> BaseTensor<RANK, NUM_T>::operator [] ( const size_t index ) const 
    requires (RANK > 1) {
        debugCheckBound(*this, index);

        VTensor<RANK-1, NUM_T> inner_view;
        for( size_t i = 0; i < RANK-1; ++i ) {
            inner_view.dimensions[i] = this->dimensions[i+1];
        }
        const size_t inner_tensor_total_size = this->total_size / this->dimensions[0];
        inner_view.total_size = inner_tensor_total_size;
        inner_view.data_ = this->data_ + (inner_tensor_total_size * index);
        return inner_view;
    }
    
    template<size_t RANK, typename NUM_T>
    VTensor<RANK-1, NUM_T> BaseTensor<RANK, NUM_T>::operator [] ( const size_t index ) 
    requires (RANK > 1) {
        // Get value from const version of this function
        const BaseTensor<RANK, NUM_T>* const_this = static_cast<const BaseTensor<RANK, NUM_T>*>(this);
        const VTensor<RANK-1, NUM_T>& const_val = const_this->operator[](index);
        return const_cast<VTensor<RANK-1, NUM_T>&>(const_val);
    }

    template<size_t RANK, typename NUM_T>
    const VTensor<RANK, NUM_T> BaseTensor<RANK, NUM_T>::slice( const size_t index_A, const size_t index_B ) const {
        debugCheckBound(*this, index_A);
        debugCheckBound(*this, index_B - 1);
        debugCheckBound(*this, index_B - 1 - index_A);

        VTensor<RANK, NUM_T> slice_view;
        // Set dimensions
        const size_t slice_size = index_B - index_A;
        slice_view.dimensions[0] = slice_size;
        for( size_t i = 1; i < RANK; ++i ) {
            slice_view.dimensions[i] = this->dimensions[i];
        }
        // Set total size
        const size_t inner_tensor_total_size = this->total_size / this->dimensions[0];
        slice_view.total_size = inner_tensor_total_size * slice_size;
        // Set inner data
        slice_view.data_ = this->data_ + (inner_tensor_total_size * index_A);
        
        return slice_view;
    }

    template<size_t RANK, typename NUM_T>
    ConstTensorTypeIterator<BaseTensor<RANK, NUM_T>> BaseTensor<RANK, NUM_T>::begin() const {
        return ConstTensorTypeIterator<BaseTensor<RANK, NUM_T>>(*this, 0);
    }

    template<size_t RANK, typename NUM_T>
    ConstTensorTypeIterator<BaseTensor<RANK, NUM_T>> BaseTensor<RANK, NUM_T>::end() const {
        return ConstTensorTypeIterator<BaseTensor<RANK, NUM_T>>(*this, this->size());
    }

    template<size_t RANK, typename NUM_T>
    TensorTypeIterator<BaseTensor<RANK, NUM_T>> BaseTensor<RANK, NUM_T>::begin() {
        return TensorTypeIterator<BaseTensor<RANK, NUM_T>>(*this, 0);
    }

    template<size_t RANK, typename NUM_T>
    TensorTypeIterator<BaseTensor<RANK, NUM_T>> BaseTensor<RANK, NUM_T>::end() {
        return TensorTypeIterator<BaseTensor<RANK, NUM_T>>(*this, this->size());
    }
 
    template<size_t RANK, typename NUM_T>
    const VTensor<RANK, NUM_T> BaseTensor<RANK, NUM_T>::view() const {
        VTensor<RANK, NUM_T> view;
        for( size_t i = 0; i < RANK; ++i ) {
            view.dimensions[i] = this->dimensions[i];
        }
        view.total_size = this->total_size;
        view.data_ = this->data_;
        return view;
    }
    
    template<size_t RANK, typename NUM_T>
    VTensor<RANK, NUM_T> BaseTensor<RANK, NUM_T>::view() {
        VTensor<RANK, NUM_T> view;
        for( size_t i = 0; i < RANK; ++i ) {
            view.dimensions[i] = this->dimensions[i];
        }
        view.total_size = this->total_size;
        view.data_ = this->data_;
        return view;
    }
    
    template<size_t RANK, typename NUM_T>
    const VTensor<RANK+1> BaseTensor<RANK, NUM_T>::rankUp() const {
        VTensor<RANK+1> ranked_up_view;
        for( size_t i = 0; i < RANK; ++i ) {
            ranked_up_view.dimensions[i] = this->dimensions[i];
        }
        ranked_up_view.dimensions[RANK] = 1;
        ranked_up_view.total_size = this->total_size;
        ranked_up_view.data_ = this->data_;
        return ranked_up_view;
    }
    
    template<size_t RANK, typename NUM_T>
    VTensor<RANK+1> BaseTensor<RANK, NUM_T>::rankUp() {
        VTensor<RANK+1> ranked_up_view;
        for( size_t i = 0; i < RANK; ++i ) {
            ranked_up_view.dimensions[i] = this->dimensions[i];
        }
        ranked_up_view.dimensions[RANK] = 1;
        ranked_up_view.total_size = this->total_size;
        ranked_up_view.data_ = this->data_;
    }
    
    template<size_t RANK, typename NUM_T>
    const VTensor<1, NUM_T> BaseTensor<RANK, NUM_T>::flattened() const {
        VTensor<1, NUM_T> flattened_view;
        flattened_view.dimensions[0] = this->total_size;
        flattened_view.total_size = this->total_size;
        flattened_view.data_ = this->data_;
        return flattened_view;
    }
    
    template<size_t RANK, typename NUM_T>
    VTensor<1, NUM_T> BaseTensor<RANK, NUM_T>::flattened() {
        VTensor<1, NUM_T> flattened_view;
        flattened_view.dimensions[0] = this->total_size;
        flattened_view.total_size = this->total_size;
        flattened_view.data_ = this->data_;
        return flattened_view;
    }

    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T> BaseTensor<RANK, NUM_T>::operator + ( const BaseTensor<RANK, NUM_T>& other ) const {
        debugCheckSizes(*this, other);

        Tensor<RANK, NUM_T> result = this->emptied();
        // Add the values in the Tensors together
        addValues(this->data_, other.data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T> BaseTensor<RANK, NUM_T>::operator - ( const BaseTensor<RANK, NUM_T>& other ) const {
        debugCheckSizes(*this, other);

        Tensor<RANK, NUM_T> result = this->emptied();
        // Subtract the values in this Tensor by the values in the other Tensor
        subtractValues(this->data_, other.data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T> BaseTensor<RANK, NUM_T>::operator * ( const BaseTensor<RANK, NUM_T>& other ) const {
        debugCheckSizes(*this, other);

        Tensor<RANK, NUM_T> result = this->emptied();
        // Multiply the values in the Tensors together
        multiplyValues(this->data_, other.data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T> BaseTensor<RANK, NUM_T>::operator / ( const BaseTensor<RANK, NUM_T>& other ) const {
        debugCheckSizes(*this, other);

        Tensor<RANK, NUM_T> result = this->emptied();
        // Divide the values in this Tensor by the values in the other Tensor
        divideValues(this->data_, other.data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T> operator * ( const BaseTensor<RANK, NUM_T>& tensor, const typename BaseTensor<RANK, NUM_T>::Num_t scale ) {
       Tensor<RANK, NUM_T> result = tensor.emptied();
       // Multiply the values in the Tensor by the scale
       multiplyValuesByScalar(tensor.data_, scale, result.data_, tensor.total_size);
    
       return result;
    }
        
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T> operator * ( const typename BaseTensor<RANK, NUM_T>::Num_t scale, const BaseTensor<RANK, NUM_T>& tensor ) {
       Tensor<RANK, NUM_T> result = tensor.emptied();
       // Multiply the values in the Tensor by the scale
       multiplyValuesByScalar(tensor.data_, scale, result.data_, tensor.total_size);
       
       return result;
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T> BaseTensor<RANK, NUM_T>::operator / ( const NUM_T scale ) const {
        Tensor<RANK, NUM_T> result = this->emptied();
        // Multiply the values in this Tensor by the inverted scale
        multiplyValuesByScalar(this->data_, (1.0f / scale), result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T> BaseTensor<RANK, NUM_T>::operator - () const {
        Tensor<RANK, NUM_T> result = this->emptied();
        // Negate the values in this Tensor
        negateValues(this->data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T> BaseTensor<RANK, NUM_T>::emptied() const {
        return Tensor<RANK, NUM_T>(this->dimensions);
    } 

    template<size_t RANK, typename NUM_T>
    bool BaseTensor<RANK, NUM_T>::isSameSize( const BaseTensor<RANK, NUM_T>& other ) const {
        for( size_t i = 0; i < RANK; ++i ) {
            if( this->dimensions[i] != other.dimensions[i] ) {
                return false;
            }
        }
        return true;
    }
    
    template<size_t RANK, typename NUM_T>
    bool BaseTensor<RANK, NUM_T>::operator == ( const BaseTensor<RANK, NUM_T>& other ) const {
        if( this == &other ) {
            return true;
        }
        if( !this->isSameSize(other) ) {
            return false;
        }
        return compareValuesForEquality(this->data_, other.data_, this->total_size);
    }
    
    template<size_t RANK, typename NUM_T>
    bool BaseTensor<RANK, NUM_T>::operator != ( const BaseTensor<RANK, NUM_T>& other ) const {
        return !(*this == other);
    }

    template<size_t RANK, typename NUM_T>
    BaseTensor<RANK, NUM_T>& BaseTensor<RANK, NUM_T>::operator += ( const BaseTensor<RANK, NUM_T>& other ) {
        debugCheckSizes(*this, other);

        // Add other's values
        addValues(this->data_, other.data_, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    BaseTensor<RANK, NUM_T>& BaseTensor<RANK, NUM_T>::operator -= ( const BaseTensor<RANK, NUM_T>& other ) {
        // Subtract other's values
        subtractValues(this->data_, other.data_, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    BaseTensor<RANK, NUM_T>& BaseTensor<RANK, NUM_T>::operator *= ( const BaseTensor<RANK, NUM_T>& other ) {
        debugCheckSizes(*this, other);

        // Multiply other's values
        multiplyValues(this->data_, other.data_, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    BaseTensor<RANK, NUM_T>& BaseTensor<RANK, NUM_T>::operator /= ( const BaseTensor<RANK, NUM_T>& other ) {
        debugCheckSizes(*this, other);

        // Divide other's values
        divideValues(this->data_, other.data_, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    BaseTensor<RANK, NUM_T>& BaseTensor<RANK, NUM_T>::operator *= ( const NUM_T scale ) {
        // Multiply by scale
        multiplyValuesByScalar(this->data_, scale, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    BaseTensor<RANK, NUM_T>& BaseTensor<RANK, NUM_T>::operator /= ( const NUM_T scale ) {
        // Divide by scale
        multiplyValuesByScalar(this->data_, 1.0f / scale, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    BaseTensor<RANK, NUM_T>& BaseTensor<RANK, NUM_T>::fill( const NUM_T fill ) {
        fillValues(fill, this->data, this->total_size);
        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    BaseTensor<RANK, NUM_T>& BaseTensor<RANK, NUM_T>::set( const BaseTensor<RANK, NUM_T>& other ) {
        debugCheckSizes(*this, other);

        setValues(other.data_, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    template<typename Func>
    BaseTensor<RANK, NUM_T>& BaseTensor<RANK, NUM_T>::transform( Func transform_function ) {
        // Define a types for the value and index, so that the value is passed when
        // `NUM_T` is defined as the argument and so that the index is passed when 
        // `size_t` is defined as the argument.
        // Since `NUM_T` and `size_t` can be implicitly converted between each other
        // these types are needed to differentiate between a `NUM_T` and `size_t`
        // argument in `transform_function`.
        struct Value_t {
            operator NUM_T();
            operator size_t() = delete;
        };
        struct Index_t {
            operator size_t();
            operator NUM_T() = delete;
        };
        
        // If the function takes no arguments
        if constexpr( std::is_invocable_r_v<NUM_T, Func> ) {
            for( size_t i = 0; i < this->total_size; ++i ) {
                this->data_[i] = transform_function();
            }
            return *this;
        }

        // If the function just takes the values, but not any indexes
        else if constexpr( std::is_invocable_r_v<NUM_T, Func, Value_t> ) {
            for( size_t i = 0; i < this->total_size; ++i ) {
                this->data_[i] = transform_function(this->data_[i]);
            }
            return *this;
        }
        
        // For RANK=1 Tensors
        else if constexpr( RANK == 1 ) {
            for( size_t i = 0; i < this->total_size; ++i ) {
                // If the function just accepts an index
                if constexpr( std::is_invocable_r_v<NUM_T, Func, Index_t> ) {
                    this->data_[i] = transform_function(i);
                }

                // If the function accepts an index and then a tensor value
                else if constexpr( std::is_invocable_r_v<NUM_T, Func, Index_t, Value_t> ) {
                    this->data_[i] = transform_function(i, this->data_[i]);
                }

                // If the function accepts a tensor value and then an index
                else if constexpr( std::is_invocable_r_v<NUM_T, Func, Value_t, Index_t> ) { 
                    this->data_[i] = transform_function(this->data_[i], i);
                }

                // Don't compile if the function doesn't match any expected formats
                else {
                    static_assert(
                        false, 
                        "The transform_function does not match any expected formats."
                    );
                }
            }
            return *this;
        }

        // For RANK>1 Tensors
        else if constexpr( RANK > 1 ) {
            // Start the indexes at 0
            size_t indexes[RANK];
            for( size_t i = 0; i < RANK; ++i ) {
                indexes[i] = 0;
            }

            while( true ) {
                // If the function just accepts an index
                if constexpr( std::is_invocable_r_v<NUM_T, Func, size_t[RANK]> ) {
                    (*this)[indexes] = transform_function(indexes);
                }

                // If the function accepts an index and then a tensor value
                else if constexpr( std::is_invocable_r_v<NUM_T, Func, size_t[RANK], NUM_T> ) {
                    (*this)[indexes] = transform_function(indexes, (*this)[indexes]);
                }

                // If the function accepts a tensor value and then an index
                else if constexpr( std::is_invocable_r_v<NUM_T, Func, NUM_T, size_t[RANK]> ) {
                    (*this)[indexes] = transform_function((*this)[indexes], indexes);
                }

                // Don't compile if the function doesn't match any expected formats
                else {
                    static_assert(
                        false, 
                        "The transform_function does not match any expected formats."
                    );
                }

                // Step the index forward
                indexes[RANK - 1]++;

                // If the index in any dimension is at the max, increment the previous dimension
                size_t dimension = RANK - 1;
                while( indexes[dimension] >= this->dimensions[dimension] ) {
                    // Stop incrementing previous dimensions when we reach the first one
                    if( dimension == 0 ) {
                        // If the first dimension has reached its maximum, iteration is finished
                        if( indexes[0] == this->dimensions[0] ) {
                            return *this;
                        }
                        break;
                    }

                    indexes[dimension] = 0;
                    indexes[dimension - 1] ++;
                    dimension--;
                }
            }
            return *this;
        }
    }

    template<size_t RANK, typename NUM_T>
    NUM_T BaseTensor<RANK, NUM_T>::mag() const 
    requires (RANK == 1) {
        return std::sqrt(this->squaredMag());
    }
    
    template<size_t RANK, typename NUM_T>
    NUM_T BaseTensor<RANK, NUM_T>::squaredMag() const 
    requires (RANK == 1) {
        NUM_T sqrd_sum = 0;
        for( size_t i = 0; i < this->total_size; ++i ) {
            sqrd_sum += this->data_[i] * this->data_[i];
        }
        return sqrd_sum;
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<1, NUM_T> BaseTensor<RANK, NUM_T>::normalized() const 
    requires (RANK == 1) {
        const NUM_T mag = this->mag();
        return (*this) / mag;
    }
    
    template<size_t RANK, typename NUM_T>
    NUM_T BaseTensor<RANK, NUM_T>::dot( const BaseTensor<1, NUM_T>& other ) const 
    requires (RANK == 1) {
        debugCheckSizes(*this, other);

        NUM_T sum = 0;
        for( size_t i = 0; i < this->total_size; ++i ) {
            sum += this->data_[i] * other.data_[i];
        }
        return sum;
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<1, NUM_T> BaseTensor<RANK, NUM_T>::cross( const BaseTensor<1, NUM_T>& other ) const 
    requires (RANK == 1) {
        debugCheckCrossProduct(*this, other);

        Tensor<1, NUM_T> result(3);
        result[0] = this[1] * other[2] - this[2] * other[1];
        result[1] = this[2] * other[0] - this[0] * other[2];
        result[2] = this[0] * other[1] - this[1] * other[0];
        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    NUM_T BaseTensor<RANK, NUM_T>::mean() const {
        NUM_T sum = 0;
        for( size_t i = 0; i < this->total_size; ++i ) {
            sum += this->data_[i];
        }
        return sum / this->total_size;
    }

    template<size_t RANK, typename NUM_T>
    Tensor<2, NUM_T> BaseTensor<RANK, NUM_T>::transpose() const 
    requires (RANK == 2) {
        Tensor<2, NUM_T> result({this->dimensions[1], this->dimensions[0]});
        for( size_t i = 0; i < this->dimensions[0]; ++i ) {
            for( size_t j = 0; j < this->dimensions[1]; ++j ) {
                result[{j, i}] = (*this)[{i, j}];
            }
        }
        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<2, NUM_T> BaseTensor<RANK, NUM_T>::transpose() const 
    requires (RANK == 1) {
        Tensor<2, NUM_T> result({1, this->dimensions[0]});
        setValues(this->data_, result.data_, this->total_size);
        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<2, NUM_T> BaseTensor<RANK, NUM_T>::mul( const BaseTensor<2, NUM_T>& other ) const 
    requires (RANK == 2) {
        debugCheckMatrixMult(*this, other);

        // Create result Tensor
        Tensor<2, NUM_T> result({this->dimensions[0], other.dimensions[1]});
        // Perform matrix multiplication
        for( size_t i = 0; i < result.dimensions[0]; ++i ) {
            for( size_t j = 0; j < result.dimensions[1]; ++j ) {
                NUM_T sum = 0;
                for( size_t k = 0; k < this->dimensions[1]; ++k ) {
                    sum += (*this)[{i, k}] * other[{k, j}];
                }
                result[{i, j}] = sum;
            }
        }
        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<1, NUM_T> BaseTensor<RANK, NUM_T>::mul( const BaseTensor<1, NUM_T>& other ) const 
    requires (RANK == 2) {
        debugCheckMatrixMult(*this, other);

        // Create result Tensor
        Tensor<1, NUM_T> result(this->dimensions[0]);
        // Perform matrix multiplication
        for( size_t i = 0; i < this->dimensions[0]; ++i ) {
            NUM_T sum = 0;
            for( size_t j = 0; j < this->dimensions[1]; ++j ) {
                sum += (*this)[{i, j}] * other[j];
            }
            result[i] = sum;
        }
        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<2, NUM_T> BaseTensor<RANK, NUM_T>::mul( const BaseTensor<2, NUM_T>& other ) const 
    requires (RANK == 1) {
        debugCheckMatrixMult(*this, other);

        // Create result Tensor
        Tensor<2, NUM_T> result({this->dimensions[0], other.dimensions[1]});
        // Perform matrix multiplication
        for( size_t i = 0; i < result.dimensions[0]; ++i ) {
            for( size_t j = 0; j < result.dimensions[1]; ++j ) {
                result[{i, j}] = (*this)[i] * other[{0, j}];
            }
        }
        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    NUM_T BaseTensor<RANK, NUM_T>::determinant() const 
    requires (RANK == 2) {
        debugCheckDeterminant(*this);

        const size_t size = this->dimensions[0];
        if( size == 2 ) {
            return this->data_[0] * this->data_[3] - this->data_[1] * this->data_[2];
        }

        NUM_T determinant = 0;
        NUM_T sign = -1;
        for( size_t k = 0; k < size; ++k ) {
            sign *= -1;

            // Get coefficient
            const NUM_T value_0_i = (*this)[{0, k}];
            // If the value is 0, skip finding determinant
            if( value_0_i == 0.0f ) {
                continue;
            }

            // Create sub matrix
            Tensor<2, NUM_T> sub_matrix({size - 1, size - 1});
            // Copy over data
            for( size_t i = 0; i < size-1; ++i ) {
                for( size_t j = 0, this_j = 0; j < size-1; ++j, ++this_j ) {
                    if( this_j == k ) {
                        ++this_j;
                    }
                    sub_matrix[{i, j}] = (*this)[{i + 1, this_j}];
                }
            }
            // Add determinant
            determinant += sign * value_0_i * sub_matrix.determinant();
        }
        
        return determinant;
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<2, NUM_T> BaseTensor<RANK, NUM_T>::inverse() const 
    requires (RANK == 2) {
        Tensor<2, NUM_T> this_copy = this->transpose();
        Tensor<2, NUM_T> result = Tensor<2, NUM_T>::identity(this->dimensions[0]);

        const size_t size = this->dimensions[0];
        for( size_t i = 0; i < size; ++i ) {
            // Divide ith row by ith value in row
            const NUM_T diagonal_i = this_copy[{i, i}];
            if ( diagonal_i == 0 ) {
                break;
            }
            const NUM_T scale_i = 1.0f / diagonal_i;
            this_copy[i] *= scale_i;
            result[i] *= scale_i;

            // Delete values in ith column below this row
            for( size_t j = i + 1; j < size; ++j ) {
                const NUM_T sub_row_scale = this_copy[{j, i}];
                this_copy[j] -= this_copy[i] * sub_row_scale;
                result[j] -= result[i] * sub_row_scale;
            }

            // Delete values in ith column above this row
            for( size_t j = 0; j < i; ++j ) {
                const Tensor<1, NUM_T> row_sub = this_copy[i] * this_copy[{j, i}];
                const NUM_T sub_row_scale = this_copy[{j, i}];
                this_copy[j] -= this_copy[i] * sub_row_scale;
                result[j] -= result[i] * sub_row_scale;
            }
        }

        return result.transpose();
    }

    template<size_t RANK, typename NUM_T>
    constexpr size_t BaseTensor<RANK, NUM_T>::rank() const {
        return RANK;
    }
    
    template<size_t RANK, typename NUM_T>
    size_t BaseTensor<RANK, NUM_T>::totalSize() const {
        return this->total_size;
    }
    
    template<size_t RANK, typename NUM_T>
    const NUM_T* BaseTensor<RANK, NUM_T>::data() const {
        return this->data_;
    }
    
    template<size_t RANK, typename NUM_T>
    size_t BaseTensor<RANK, NUM_T>::size() const {
        return this->dimensions[0];
    }
    
    template<size_t RANK, typename NUM_T>
    size_t BaseTensor<RANK, NUM_T>::size( const size_t dim_index ) const {
        debugCheckDimensionsBound(*this, dim_index);

        return this->dimensions[dim_index];
    }

    template<size_t RANK, typename NUM_T>
    std::ostream& operator << ( std::ostream& fs, const BaseTensor<RANK, NUM_T>& t ) {
        // Open Tensor
        fs << "{ ";
        // Print inner Tensors
        if( t.dimensions[0] > 0 ) fs << t[0];
        for( size_t i = 1; i < t.dimensions[0]; ++i ) {
            fs << ", ";
            fs << t[i];
        }
        // Close Tensor
        fs << " }";
        return fs;
    }


    /* Implementation Helper Functions For Handling Recursive std::initializer_lists */

    namespace {
        
        template<size_t RANK, typename NUM_T>
        size_t countInitializerElements( const InitializerElements<RANK, NUM_T>& elements, size_t dims[RANK] ) {
            dims[0] = elements.size();

            // RANK=1 case
            if constexpr( RANK == 1 ) {
                return elements.size();
            }

            // RANK>1 case
            else {
                const size_t inner_size = countInitializerElements<RANK-1, NUM_T>(*elements.begin(), dims + 1);
                return elements.size() * inner_size;
            }
        }
        
        template<size_t RANK, typename NUM_T>
        bool checkInitializerElements( const InitializerElements<RANK, NUM_T>& elements, const size_t dims[RANK] ) {
            if( elements.size() != dims[0] ) {
                return false;
            }

            // RANK=1 case
            if constexpr( RANK == 1 ) {
                return true;
            }

            // RANK>1 case
            else {
                for( const auto& inner_elements : elements ) {
                    if( !checkInitializerElements<RANK-1, NUM_T>(inner_elements, dims + 1) ) {
                        return false;
                    }
                }
                return true;
            }
        }
        
        template<size_t RANK, typename NUM_T>
        void flattenInitializerElements( const InitializerElements<RANK, NUM_T>& elements, NUM_T*& data ) {
            // RANK=1 case
            if constexpr( RANK == 1 ) {
                for( const NUM_T element : elements ) {
                    *data = element;
                    ++data;
                }
            }

            // RANK>1 case
            else {
                for( const auto& inner_elements : elements ) {
                    flattenInitializerElements<RANK-1, NUM_T>(inner_elements, data);
                }
            }
        }
    }
    

    /* Tensor Implementation */

    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T>::Tensor() {
        // Set all dimensions to 0
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = 0;
        }
        // Allocate no memory
        this->total_size = 0;
        this->data_ = nullptr;
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T>::Tensor( const size_t dim ) 
    requires (RANK == 1) {
        if( dim == 0 ) {
            throw std::invalid_argument("The dimension size is less than 1.");
        }
        this->dimensions[0] = dim;
        this->total_size = dim;
        this->data_ = new NUM_T[dim];
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T>::Tensor( const size_t dim, const NUM_T fill ) 
    requires (RANK == 1) 
    : Tensor(dim) {
        fillValues(fill, this->data_, this->total_size);
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T>::Tensor( size_t dim, const NUM_T fill[] )
    requires (RANK == 1) 
    : Tensor(dim) {
        setValues(fill, this->data_, dim);
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T>::Tensor( const size_t (&dims)[RANK] ) {
        // Copy dimensions
        size_t total_size = 1;
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = dims[i];
            total_size *= dims[i];
            // Check if the size of this dimension is 0
            if( dims[i] < 1 ) {
                throw std::invalid_argument("One or more dimension sizes are less than 1.");
            }
        }
        // Allocate memory for data
        this->total_size = total_size;
        this->data_ = new NUM_T[total_size];
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T>::Tensor( const size_t (&dims)[RANK], const NUM_T fill ) 
    : Tensor(dims) {
        fillValues(fill, this->data_, this->total_size);
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T>::Tensor( InitializerElements<RANK, NUM_T> elements ) {
        if( elements.size() == 0 ) {
            throw std::invalid_argument("The first dimension size is less than 1.");
        }

        // Recursively count the total size of the initializer elements
        this->total_size = countInitializerElements<RANK, NUM_T>(elements, this->dimensions);
        if( !checkInitializerElements<RANK, NUM_T>(elements, this->dimensions) ) {
            throw std::invalid_argument("The given initializer elements are not rectangular");
        }

        // Check if any of the dimensions are 0
        for( size_t i = 0; i < RANK; ++i ) {
            if( this->dimensions[i] < 1 ) {
                throw std::invalid_argument("One or more dimension sizes are less than 1.");
            }
        }

        // Allocate memory
        this->data_ = new NUM_T[this->total_size];
        // Assign data from flattened initializer elements
        NUM_T* data_ptr = this->data_;
        flattenInitializerElements<RANK, NUM_T>(elements, data_ptr);
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T>::Tensor( std::initializer_list<std::reference_wrapper<const BaseTensor<RANK-1, NUM_T>>> elements ) 
    requires (RANK > 1) {
        const size_t dim1 = elements.size();
        if( dim1 == 0 ) {
            throw std::invalid_argument("The first dimension size is less than 1.");
        }

        const std::reference_wrapper<const BaseTensor<RANK-1, NUM_T>>* tensor_refs = elements.begin();
        // Copy dimensions from the first Tensor
        this->dimensions[0] = dim1;
        for( size_t i = 1; i < RANK; ++i ) {
            this->dimensions[i] = tensor_refs[0].get().dimensions[i-1];
        }
        // Check that all Tensors have the same dimensions
        for( size_t i = 1; i < dim1; ++i ) {
            for( size_t j = 0; j < RANK-1; ++j ) {
                if( this->dimensions[j+1] != tensor_refs[i].get().dimensions[j] ) {
                    throw std::invalid_argument("Two or more dimension sizes do not match.");
                }
            }
        }
        // Allocate memory for data
        const size_t inner_tensor_size = tensor_refs[0].get().total_size;
        this->total_size = dim1 * inner_tensor_size;
        this->data_ = new NUM_T[this->total_size];
        // Copy data from Tensors into this
        for( size_t i = 0; i < dim1; ++i ) {
            setValues(tensor_refs[i].get().data_, this->data_ + (i * inner_tensor_size), inner_tensor_size);  
        }
    }

    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T>::Tensor( const std::vector<NUM_T>& vec )
    requires (RANK == 1) 
    : Tensor<1, NUM_T>(vec.size(), vec.data()) { }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T>::Tensor( const BaseTensor<RANK, NUM_T>& other ) {
        // Copy dimensions
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        // Allocate new memory and copy it over
        this->total_size = other.total_size;
        this->data_ = new NUM_T[other.total_size];
        setValues(other.data_, this->data_, this->total_size);
    }

    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T>::Tensor( const Tensor<RANK, NUM_T>& other )
    : Tensor((const BaseTensor<RANK, NUM_T>&) other) { }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T>::Tensor( Tensor<RANK, NUM_T>&& other ) {
        // Copy dimensions
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        // Set other's data pointer to this
        this->total_size = other.total_size;
        this->data_ = other.data_;

        // Clear other tensor
        for( size_t i = 0; i < RANK; ++i ) {
            other.dimensions[i] = 0;
        }
        other.total_size = 0;
        other.data_ = nullptr;
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T>::~Tensor() {
        delete[] this->data_;
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T>& Tensor<RANK, NUM_T>::operator = ( const BaseTensor<RANK, NUM_T>& other ) {
        // Check for self assignment
        if( this == &other ) {
            return *this;
        }

        // Copy dimensions
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        // Free the previous data held in this Tensor, if a different size
        if( this->total_size != other.total_size ) {
            delete[] this->data_;
            this->data_ = new NUM_T[other.total_size];
            this->total_size = other.total_size;
        }
        // Copy over data
        setValues(other.data_, this->data_, this->total_size);

        return *this;
    }

    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T>& Tensor<RANK, NUM_T>::operator = ( const Tensor<RANK, NUM_T>& other ) {
        this->operator=((const BaseTensor<RANK, NUM_T>&) other);
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<RANK, NUM_T>& Tensor<RANK, NUM_T>::operator = ( Tensor<RANK, NUM_T>&& other ) {
        // Check for self assignment
        if( this == &other ) {
            return *this;
        }
        // Free the previous data held in this Tensor.
        delete[] this->data_;

        // Copy dimensions
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        // Set other's data pointer to this
        this->total_size = other.total_size;
        this->data_ = other.data_;

        // Clear other tensor
        for( size_t i = 0; i < RANK; ++i ) {
            other.dimensions[i] = 0;
        }
        other.total_size = 0;
        other.data_ = nullptr;

        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    Tensor<1, NUM_T> Tensor<RANK, NUM_T>::range( const size_t dim, const NUM_T min, const NUM_T max ) {
        Tensor<1, NUM_T> result(dim);
        // Determine spacing
        NUM_T range = max - min;
        NUM_T step = range / (dim - 1);
        for( size_t i = 0; i < dim; ++i ) {
            result[i] = min + (step * i);
        }
        return result;
    }

    template<size_t RANK, typename NUM_T>
    Tensor<2, NUM_T> Tensor<RANK, NUM_T>::identity( const size_t dims, const NUM_T diagonal_value ) {
        Tensor<2, NUM_T> result({dims, dims}, 0);
        for( size_t i = 0; i < dims; ++i ) {
            result[{i, i}] = diagonal_value;
        }
        return result;
    }


    /* VTensor Implementation */

    template<size_t RANK, typename NUM_T>
    VTensor<RANK, NUM_T>::VTensor() {
        // Set all dimensions to 0
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = 0;
        }
        // Allocate no memory
        this->total_size = 0;
        this->data_ = nullptr;
    }
    
    template<size_t RANK, typename NUM_T>
    VTensor<RANK, NUM_T>::VTensor( const VTensor<RANK, NUM_T>& other ) {
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        this->total_size = other.total_size;
        this->data_ = other.data_;
    }
    
    template<size_t RANK, typename NUM_T>
    VTensor<RANK, NUM_T>& VTensor<RANK, NUM_T>::operator = ( const VTensor<RANK, NUM_T>& other ) {
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        this->total_size = other.total_size;
        this->data_ = other.data_;

        return *this;
    }


    /* RaggedTensor Implementation */

    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>::RaggedTensor() {
        this->total_size = 0;
        this->data_ = nullptr;
        this->dimension1 = 0;
        this->inner_tensors = nullptr;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>::RaggedTensor( const size_t dim1_size, const size_t inner_tensor_dims[] ) 
    requires (RANK == 2) {
        if( dim1_size == 0 ) {
            throw std::invalid_argument("The first dimension size is less than 1.");
        }
        // Allocate space for array of inner VTensors
        this->inner_tensors = new VTensor<1, NUM_T>[dim1_size];
        // Copy over dimension sizes and count the total size
        size_t total_size = 0;
        for( size_t i = 0; i < dim1_size; ++i ) {
            const size_t inner_tensor_size = inner_tensor_dims[i];
            if( inner_tensor_size == 0 ) {
                throw std::invalid_argument("One or more inner dimension sizes are less than 1.");
            }
            // Copy over dimension size
            this->inner_tensors[i].dimensions[0] = inner_tensor_size;
            // Set inner tensor total size
            this->inner_tensors[i].total_size = inner_tensor_size;

            total_size += inner_tensor_size;
        }
        this->total_size = total_size;

        // Allocate space for data
        this->data_ = new NUM_T[total_size];

        // Set first dimension size
        this->dimension1 = dim1_size;
        // Set the starting position of each inner Tensor
        NUM_T* starting_pos = this->data_;
        for( size_t i = 0; i < dim1_size; ++i ) {
            this->inner_tensors[i].data_ = starting_pos;
            starting_pos += this->inner_tensors[i].total_size;
        }
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>::RaggedTensor( const size_t dim1_size, const size_t inner_tensor_dims[][RANK-1] ) {
        if( dim1_size == 0 ) {
            throw std::invalid_argument("The first dimension size is less than 1.");
        }
        // Allocate space for array of inner VTensors
        this->inner_tensors = new VTensor<RANK-1, NUM_T>[dim1_size];
        // Copy over dimension sizes and count the total size
        size_t total_size = 0;
        for( size_t i = 0; i < dim1_size; ++i ) {
            size_t inner_tensor_size = 1;
            // Copy over dimension sizes
            for( size_t j = 0; j < RANK-1; ++j ) {
                const size_t dim = inner_tensor_dims[i][j];
                if( dim == 0 ) {
                    throw std::invalid_argument("One or more inner dimension sizes are less than 1.");
                }
                this->inner_tensors[i].dimensions[j] = dim;
                inner_tensor_size *= dim;
            }
            // Set inner tensor total size
            this->inner_tensors[i].total_size = inner_tensor_size;

            total_size += inner_tensor_size;
        }
        this->total_size = total_size;

        // Allocate space for data
        this->data_ = new NUM_T[total_size];

        // Set first dimension size
        this->dimension1 = dim1_size;
        // Set the starting position of each inner Tensor
        NUM_T* starting_pos = this->data_;
        for( size_t i = 0; i < dim1_size; ++i ) {
            this->inner_tensors[i].data_ = starting_pos;
            starting_pos += this->inner_tensors[i].total_size;
        }
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>::RaggedTensor( std::initializer_list<size_t> inner_tensor_dims )
    requires (RANK == 2)  
        : RaggedTensor<RANK, NUM_T>(inner_tensor_dims.size(), inner_tensor_dims.begin()) { }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>::RaggedTensor( std::initializer_list<size_t[RANK-1]> inner_tensor_dims )
        : RaggedTensor<RANK, NUM_T>(inner_tensor_dims.size(), inner_tensor_dims.begin()) { }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>::RaggedTensor( InitializerElements<RANK, NUM_T> elements ) {
        const size_t dim1 = elements.size();
        const auto inner_elements = elements.begin();
        if( dim1 == 0 ) {
            throw std::invalid_argument("The first dimension size is less than 1.");
        }

        // Allocate space for array of inner VTensors
        this->inner_tensors = new VTensor<RANK-1, NUM_T>[dim1];
        // Get dimension sizes and count the total size
        this->total_size = 0;
        for( size_t k = 0; k < dim1; ++k ) {
            VTensor<RANK-1, NUM_T>& inner_tensor = this->inner_tensors[k];
            // Recursively count the total size of the inner initializer elements
            inner_tensor.total_size = countInitializerElements<RANK-1, NUM_T>(inner_elements[k], inner_tensor.dimensions);
            this->total_size += inner_tensor.total_size;
            // Check if the inner tensor is rectangular
            if( !checkInitializerElements<RANK-1, NUM_T>(inner_elements[k], inner_tensor.dimensions) ) {
                throw std::invalid_argument("The given inner initializer elements are not rectangular");
            }
            // Check if any of the dimensions are 0
            for( size_t i = 0; i < RANK-1; ++i ) {
                if( inner_tensor.dimensions[i] < 1 ) {
                    throw std::invalid_argument("One or more dimension sizes are less than 1.");
                }
            }
        }
        
        // Allocate memory
        this->data_ = new NUM_T[this->total_size];

        // Set first dimension size
        this->dimension1 = dim1;
        // Assign data from flattened inner initializer elements
        NUM_T* data_ptr = this->data_;
        for( size_t k = 0; k < dim1; ++k ) {
            VTensor<RANK-1, NUM_T>& inner_tensor = this->inner_tensors[k];
            // Set inner tensor elements
            inner_tensor.data_ = data_ptr;
            flattenInitializerElements<RANK-1, NUM_T>(inner_elements[k], data_ptr);
        }
    }

    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>::RaggedTensor( std::initializer_list<std::reference_wrapper<const BaseTensor<RANK-1, NUM_T>>> elements ) {
        const size_t dim1 = elements.size();
        if( dim1 == 0 ) {
            throw std::invalid_argument("The first dimension size is less than 1.");
        }

        // Allocate space for array of inner VTensors
        this->inner_tensors = new VTensor<RANK-1, NUM_T>[dim1];
        // Copy over dimension sizes and count the total size
        const std::reference_wrapper<const BaseTensor<RANK-1, NUM_T>>* inner_tensor_ptrs = elements.begin();
        size_t total_size = 0;
        for( size_t i = 0; i < dim1; ++i ) {
            // Copy over dimension sizes
            for( size_t j = 0; j < RANK-1; ++j ) {
                const size_t dim = inner_tensor_ptrs[i].get().dimensions[j];
                if( dim == 0 ) {
                    throw std::invalid_argument("One or more inner dimension sizes are less than 1.");
                }
                this->inner_tensors[i].dimensions[j] = dim;
            }
            // Set inner tensor total size
            const size_t inner_tensor_size = inner_tensor_ptrs[i].get().total_size;
            this->inner_tensors[i].total_size = inner_tensor_size;

            total_size += inner_tensor_size;
        }
        this->total_size = total_size;

        // Allocate space for data
        this->data_ = new NUM_T[total_size];

        // Set first dimension size
        this->dimension1 = dim1;
        // Set the starting position of each inner Tensor
        NUM_T* starting_pos = this->data_;
        for( size_t i = 0; i < dim1; ++i ) {
            this->inner_tensors[i].data_ = starting_pos;
            this->inner_tensors[i].set(inner_tensor_ptrs[i].get());

            starting_pos += this->inner_tensors[i].total_size;
        }
    }

    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>::RaggedTensor( const RaggedTensor<RANK, NUM_T>& other ) {
        this->total_size = other.total_size;
        // Allocate space for data and copy over
        this->data_ = new NUM_T[other.total_size];
        setValues(other.data_, this->data_, this->total_size);
        
        const size_t dim1 = other.dimension1;
        this->dimension1 = dim1;
        // Allocate space for inner tensors and copy over
        this->inner_tensors = new VTensor<RANK-1, NUM_T>[dim1];
        NUM_T* starting_pos = this->data_;
        for( size_t i = 0; i < dim1; ++i ) {
            this->inner_tensors[i] = other.inner_tensors[i];
            // Set a pointer from the newly allocated data
            this->inner_tensors[i].data_ = starting_pos;
            starting_pos += this->inner_tensors[i].total_size;
        }
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>::RaggedTensor( const BaseTensor<RANK, NUM_T>& other ) {
        this->total_size = other.total_size;
        // Allocate space for data and copy over
        this->data_ = new NUM_T[other.total_size];
        setValues(other.data_, this->data_, this->total_size);

        const size_t dim1 = other.dimensions[0];
        this->dimension1 = dim1;
        // Allocate space for inner tensors and copy over
        this->inner_tensors = new VTensor<RANK-1, NUM_T>[dim1];
        const size_t inner_tensor_total_size = other.total_size / dim1;
        NUM_T* starting_pos = this->data_;
        for( size_t i = 0; i < dim1; ++i ) {
            // Copy over sizes
            this->inner_tensors[i].total_size = inner_tensor_total_size;
            for( size_t j = 1; j < RANK; ++j ) {
                this->inner_tensors[i].dimensions[j-1] = other.dimensions[j];
            }

            // Set a pointer from the newly allocated data
            this->inner_tensors[i].data_ = starting_pos;
            starting_pos += this->inner_tensors[i].total_size;
        }
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>::RaggedTensor( RaggedTensor<RANK, NUM_T>&& other ) {
        // Move data to this RaggedTensor
        this->total_size = other.total_size;
        this->data_ = other.data_;
        this->dimension1 = other.dimension1;
        this->inner_tensors = other.inner_tensors;

        // Clear data from other RaggedTensor
        other.total_size = 0;
        other.data_ = nullptr;
        other.dimension1 = 0;
        other.inner_tensors = nullptr;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>::~RaggedTensor() {
        delete[] this->data_;
        delete[] this->inner_tensors;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>& RaggedTensor<RANK, NUM_T>::operator = ( const RaggedTensor<RANK, NUM_T>& other ) {
        // Check for self assignment
        if( this == &other ) {
            return *this;
        }
        // Free the previous data held in this RaggedTensor.
        delete[] this->data_;
        delete[] this->inner_tensors;

        this->total_size = other.total_size;
        // Allocate space for data and copy over
        this->data_ = new NUM_T[other.total_size];
        setValues(other.data_, this->data_, this->total_size);

        const size_t dim1 = other.dimension1;
        this->dimension1 = dim1;
        // Allocate space for inner tensors and copy over
        this->inner_tensors = new VTensor<RANK-1, NUM_T>[dim1];
        NUM_T* starting_pos = this->data_;
        for( size_t i = 0; i < dim1; ++i ) {
            this->inner_tensors[i] = other.inner_tensors[i];
            // Set a pointer from the newly allocated data
            this->inner_tensors[i].data_ = starting_pos;
            starting_pos += other.inner_tensors[i].total_size;
        }

        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>& RaggedTensor<RANK, NUM_T>::operator = ( const BaseTensor<RANK, NUM_T>& other ) {
        // Check for self assignment
        if( this == &other ) {
            return *this;
        }
        // Free the previous data held in this RaggedTensor.
        delete[] this->data_;
        delete[] this->inner_tensors;

        this->total_size = other.total_size;
        // Allocate space for data and copy over
        this->data_ = new NUM_T[other.total_size];
        setValues(other.data_, this->data_, this->total_size);

        const size_t dim1 = other.dimensions[0];
        this->dimension1 = dim1;
        // Allocate space for inner tensors and copy over
        this->inner_tensors = new VTensor<RANK-1, NUM_T>[dim1];
        const size_t inner_tensor_total_size = other.total_size / dim1;
        NUM_T* starting_pos = this->data_;
        for( size_t i = 0; i < dim1; ++i ) {
            // Copy over sizes
            this->inner_tensors[i].total_size = inner_tensor_total_size;
            for( size_t j = 1; j < RANK; ++j ) {
                this->inner_tensors[i].dimensions[j-1] = other.dimensions[j];
            }

            // Set a pointer from the newly allocated data
            this->inner_tensors[i].data_ = starting_pos;
            starting_pos += this->inner_tensors[i].total_size;
        }

        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>& RaggedTensor<RANK, NUM_T>::operator = ( RaggedTensor<RANK, NUM_T>&& other ) {
        // Check for self assignment
        if( this == &other ) {
            return *this;
        }
        // Free the previous data held in this RaggedTensor.
        delete[] this->data_;
        delete[] this->inner_tensors;

        // Move data to this RaggedTensor
        this->total_size = other.total_size;
        this->data_ = other.data_;
        this->dimension1 = other.dimension1;
        this->inner_tensors = other.inner_tensors;

        // Clear data from other RaggedTensor
        other.total_size = 0;
        other.data_ = nullptr;
        other.dimension1 = 0;
        other.inner_tensors = nullptr;

        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    const NUM_T& RaggedTensor<RANK, NUM_T>::operator [] ( const size_t (&indexes)[RANK] ) const {
        debugCheckBound(*this, indexes);

        return this->inner_tensors[indexes[0]][tail(indexes)];
    }
    
    template<size_t RANK, typename NUM_T>
    NUM_T& RaggedTensor<RANK, NUM_T>::operator [] ( const size_t (&indexes)[RANK] ) {
        debugCheckBound(*this, indexes);

        return this->inner_tensors[indexes[0]][tail(indexes)];
    }
    
    template<size_t RANK, typename NUM_T>
    const VTensor<RANK-1, NUM_T> RaggedTensor<RANK, NUM_T>::operator [] ( size_t index ) const {
        debugCheckBound(*this, index);

        return this->inner_tensors[index];
    }
    
    template<size_t RANK, typename NUM_T>
    VTensor<RANK-1, NUM_T> RaggedTensor<RANK, NUM_T>::operator [] ( size_t index ) {
        debugCheckBound(*this, index);
        
        return this->inner_tensors[index];
    }

    template<size_t RANK, typename NUM_T>
    ConstTensorTypeIterator<RaggedTensor<RANK, NUM_T>> RaggedTensor<RANK, NUM_T>::begin() const {
        return ConstTensorTypeIterator<RaggedTensor<RANK, NUM_T>>(*this, 0);
    }

    template<size_t RANK, typename NUM_T>
    ConstTensorTypeIterator<RaggedTensor<RANK, NUM_T>> RaggedTensor<RANK, NUM_T>::end() const {
        return ConstTensorTypeIterator<RaggedTensor<RANK, NUM_T>>(*this, this->size());
    }

    template<size_t RANK, typename NUM_T>
    TensorTypeIterator<RaggedTensor<RANK, NUM_T>> RaggedTensor<RANK, NUM_T>::begin() {
        return TensorTypeIterator<RaggedTensor<RANK, NUM_T>>(*this, 0);
    }

    template<size_t RANK, typename NUM_T>
    TensorTypeIterator<RaggedTensor<RANK, NUM_T>> RaggedTensor<RANK, NUM_T>::end() {
        return TensorTypeIterator<RaggedTensor<RANK, NUM_T>>(*this, this->size());
    }
 
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T> RaggedTensor<RANK, NUM_T>::operator + ( const RaggedTensor<RANK, NUM_T>& other ) const {
        debugCheckSizes(*this, other);

        RaggedTensor<RANK, NUM_T> result = this->emptied();
        // Add the values in the Tensors together
        addValues(this->data_, other.data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T> RaggedTensor<RANK, NUM_T>::operator - ( const RaggedTensor<RANK, NUM_T>& other ) const {
        debugCheckSizes(*this, other);

        RaggedTensor<RANK, NUM_T> result = this->emptied();
        // Subtract the values in this Tensor by the values in the other Tensor
        subtractValues(this->data_, other.data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T> RaggedTensor<RANK, NUM_T>::operator * ( const RaggedTensor<RANK, NUM_T>& other ) const {
        debugCheckSizes(*this, other);

        RaggedTensor<RANK, NUM_T> result = this->emptied();
        // Multiply the values in the Tensors together
        multiplyValues(this->data_, other.data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T> RaggedTensor<RANK, NUM_T>::operator / ( const RaggedTensor<RANK, NUM_T>& other ) const {
        debugCheckSizes(*this, other);

        RaggedTensor<RANK, NUM_T> result = this->emptied();
        // Divide the values in this Tensor by the values in the other Tensor
        divideValues(this->data_, other.data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T> operator * ( const RaggedTensor<RANK, NUM_T>& tensor, const typename RaggedTensor<RANK, NUM_T>::Num_t scale ) {
        RaggedTensor<RANK, NUM_T> result = tensor.emptied();
        // Multiply the values in the Tensor by the scale
        multiplyValuesByScalar(tensor.data_, scale, result.data_, tensor.total_size);

        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T> operator * ( const typename RaggedTensor<RANK, NUM_T>::Num_t scale, const RaggedTensor<RANK, NUM_T>& tensor ) {
        RaggedTensor<RANK, NUM_T> result = tensor.emptied();
        // Multiply the values in the Tensor by the scale
        multiplyValuesByScalar(tensor.data_, scale, result.data_, tensor.total_size);
        
        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T> RaggedTensor<RANK, NUM_T>::operator / ( const NUM_T scale ) const {
        RaggedTensor<RANK, NUM_T> result = this->emptied();
        // Multiply the values in this Tensor by the inverted scale
        multiplyValuesByScalar(this->data_, (1.0f / scale), result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T> RaggedTensor<RANK, NUM_T>::operator - () const {
        RaggedTensor<RANK, NUM_T> result = this->emptied();
        // Negate the values in this Tensor
        negateValues(this->data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T> RaggedTensor<RANK, NUM_T>::emptied() const {
        size_t inner_tensor_dims[this->dimension1][RANK-1];
        // Copy over dimensions
        for( size_t i = 0; i < this->dimension1; ++i ) {
            for( size_t j  = 0; j < RANK-1; ++j ) {
                inner_tensor_dims[i][j] = this->inner_tensors[i].dimensions[j];
            }
        }
        return RaggedTensor<RANK, NUM_T>(this->dimension1, inner_tensor_dims);
    } 
    
    template<size_t RANK, typename NUM_T>
    bool RaggedTensor<RANK, NUM_T>::isSameSize( const RaggedTensor<RANK, NUM_T>& other ) const {
        if( this->dimension1 != other.dimension1 ) {
            return false;
        }
        for( size_t i = 0; i < this->dimension1; ++i ) {
            if( !this->inner_tensors[i].isSameSize(other.inner_tensors[i]) ) {
                return false;
            }
        }
        return true;
    }
    
    template<size_t RANK, typename NUM_T>
    bool RaggedTensor<RANK, NUM_T>::operator == ( const RaggedTensor<RANK, NUM_T>& other ) const {
        if( this == &other ) {
            return true;
        }
        if( !this->isSameSize(other) ) {
            return false;
        }
        return compareValuesForEquality(this->data_, other.data_, this->total_size);
    }
    
    template<size_t RANK, typename NUM_T>
    bool RaggedTensor<RANK, NUM_T>::operator != ( const RaggedTensor<RANK, NUM_T>& other ) const {
        return !(*this == other);
    }

    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>& RaggedTensor<RANK, NUM_T>::operator += ( const RaggedTensor<RANK, NUM_T>& other ) {
        debugCheckSizes(*this, other);

        // Add other's values
        addValues(this->data_, other.data_, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>& RaggedTensor<RANK, NUM_T>::operator -= ( const RaggedTensor<RANK, NUM_T>& other ) {
        debugCheckSizes(*this, other);

        // Subtract other's values
        subtractValues(this->data_, other.data_, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>& RaggedTensor<RANK, NUM_T>::operator *= ( const RaggedTensor<RANK, NUM_T>& other ) {
        debugCheckSizes(*this, other);

        // Add other's values
        multiplyValues(this->data_, other.data_, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>& RaggedTensor<RANK, NUM_T>::operator /= ( const RaggedTensor<RANK, NUM_T>& other ) {
        debugCheckSizes(*this, other);

        // Subtract other's values
        divideValues(this->data_, other.data_, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>& RaggedTensor<RANK, NUM_T>::operator *= ( const NUM_T scale ) {
        // Multiply by scale
        multiplyValuesByScalar(this->data_, scale, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>& RaggedTensor<RANK, NUM_T>::operator /= ( const NUM_T scale ) {
        // Multiply by scale
        multiplyValuesByScalar(this->data_, 1.0f / scale, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>& RaggedTensor<RANK, NUM_T>::fill( const NUM_T fill ) {
        fillValues(fill, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK, typename NUM_T>
    RaggedTensor<RANK, NUM_T>& RaggedTensor<RANK, NUM_T>::set( const RaggedTensor<RANK, NUM_T>& other ) {
        debugCheckSizes(*this, other);

        setValues(other.data_, this->data_, this->total_size);
        return *this;
    }
    
    /* TODO: Add support for indexing */
    template<size_t RANK, typename NUM_T>
    template<typename Func>
    RaggedTensor<RANK, NUM_T>& RaggedTensor<RANK, NUM_T>::transform( Func transform_function ) {
        // Define a types for the value and index, so that the value is passed when
        // `NUM_T` is defined as the argument and so that the index is passed when 
        // `size_t` is defined as the argument.
        // Since `NUM_T` and `size_t` can be implicitly converted between each other
        // these types are needed to differentiate between a `NUM_T` and `size_t`
        // argument in `transform_function`.
        struct Value_t {
            operator NUM_T();
            operator size_t() = delete;
        };

        // If the function takes no arguments
        if constexpr( std::is_invocable_r_v<NUM_T, Func> ) {
            for( size_t i = 0; i < this->total_size; ++i ) {
                this->data_[i] = transform_function();
            }
            return *this;
        }

        // If the function just takes the values, but not any indexes
        else if constexpr( std::is_invocable_r_v<NUM_T, Func, Value_t> ) {
            for( size_t i = 0; i < this->total_size; ++i ) {
                this->data_[i] = transform_function(this->data_[i]);
            }
            return *this;
        }

        // Don't compile if the function doesn't match any expected formats
        else {
            static_assert(
                false, 
                "The transform_function does not match any expected formats."
            );
        }
    }

    template<size_t RANK, typename NUM_T>
    constexpr size_t RaggedTensor<RANK, NUM_T>::rank() const {
        return RANK;
    }
    
    template<size_t RANK, typename NUM_T>
    size_t RaggedTensor<RANK, NUM_T>::totalSize() const {
        return this->total_size;
    }
    
    template<size_t RANK, typename NUM_T>
    size_t RaggedTensor<RANK, NUM_T>::size() const {
        return this->dimension1;
    }
    
    template<size_t RANK, typename NUM_T>
    const NUM_T* RaggedTensor<RANK, NUM_T>::data() const {
        return this->data_;
    }

    template<size_t RANK, typename NUM_T>
    std::ostream& operator << ( std::ostream& fs, const RaggedTensor<RANK, NUM_T>& rt ) {
        // Open Tensor
        fs << "{ ";
        // Print inner Tensors
        if( rt.dimension1 > 0 ) fs << rt[0];
        for( size_t i = 1; i < rt.dimension1; ++i ) {
            fs << ", ";
            fs << rt[i];
        }
        // Close Tensor
        fs << " }";
        return fs;
    }
}



#endif