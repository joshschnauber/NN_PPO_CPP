/* Tensor.hpp */

/**
 * TODO:
 * - Add optional template parameter for other types in Tensor (still float as default)
 * 
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
 * 
 * - Somehow connect Tensor and VTensor with RaggedTensor, so their overlapping 
 *   functionality can be reused, and to get rid of the need for duplicated code.
 * 
 * - Make the size of Tensors completely unchangeable by only making the assignment
 *   operator just copy over values, and not change the size of the Tensor. Need to have
 *   some way of making handling Tensors easier then, because allowing resizing on 
 *   assignment is convienient.
 *   This will also let us remove the default constructor (and maybe make .total_size 
 *   and some other size variables const?).
 * 
 * - Add .collapsed() function that collapses Tensors with rank RANK into Tensor with rank
 *   RANK-1, when RANK>1.
 * - Add iterator to allow enhanced iterate to iterate over rows of rank RANK-1, or just
 *   a float if RANK=1
 * - Add a function to fill the Tensor with random values
 * 
 * - Add .copy() functions that returns a new Tensor, and make more non size changing
 *   mutators alter the Tensor instead of creating a new one.
 */
 
#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <initializer_list>
#include <cassert>

#include <type_traits>
#include <concepts>
#include <utility>



/* Declaration */
namespace jai {
    /**
     * Helper structs to define a recursive type for initializing a Tensor with elements.
     */
    template <size_t RANK>
    struct InitializerElementsType {
        using type = std::initializer_list<typename InitializerElementsType<RANK - 1>::type>;
    };
    template <>
    struct InitializerElementsType<1> {
        using type = std::initializer_list<float>;
    };
    /**
     * Recursive type used to initialize a Tensor of rank `RANK` with elements.
     * An InitializerElements<RANK> contains a set of InitializerElements<RANK-1>s,
     * and an InitializerElements<1> contains a set of floats.
     */
    template <size_t RANK>
    using InitializerElements = typename InitializerElementsType<RANK>::type;


    /* Tensor Declarations */
    template<size_t RANK>
    class BaseTensor;
    template<size_t RANK>
    class Tensor;
    template<size_t RANK>
    class VTensor;
    template<size_t RANK>
    class RaggedTensor;


    /**
     * This defines the interface with a Tensor.
     * This is an abstract class that cannot be constructed on it's own.
     */
    template<size_t RANK>
    class BaseTensor {
        // Ensure that Tensor RANK cannot be 0 (must have 1 or more dimensions)
        static_assert(RANK > 0, "Tensor rank cannot be 0.");

        /* Constructors */
        protected:

        BaseTensor();

        /* Accessors */
        public:

        /**
         * Defined for RANK=1 Tensors, this returns the element at the given index in the
         * first (and only) dimension.
         */
        const float& operator [] ( size_t index ) const 
        requires (RANK == 1);
        /**
         * Defined for RANK=1 Tensors, this returns a mutable reference to the element at
         * the `index` in the first (and only) dimension.
         */
        float& operator [] ( size_t index )
        requires (RANK == 1);
        /**
         * Defined for RANK>1 Tensors, returns the element at the given indexes.
         */
        const float& operator [] ( const size_t (&indexes)[RANK] ) const;
        /**
         * Defined for RANK>1 Tensors, this returns a mutable reference to the element
         * at the given `indexes`.
         */
        float& operator [] ( const size_t (&indexes)[RANK] );
        /**
         * Defined for RANK>1 Tensors, this returns an immutable View Tensor with rank
         * RANK-1, at the given index in the first dimension.
         * The returned View Tensor is backed by `this` Tensor.
         */
        const VTensor<RANK-1> operator [] ( size_t index ) const
        requires (RANK > 1);
        /**
         * Defined for RANK>1 Tensors, this returns a View Tensor with rank RANK-1, at
         * the given index in the first dimension.
         * The returned View Tensor is backed by `this` Tensor.
         */
        VTensor<RANK-1> operator [] ( size_t index )
        requires (RANK > 1);

        /**
         * Returns an immutable View Tensor which is backed by `this` Tensor.
         */
        const VTensor<RANK> view() const;
        /**
         * Returns a View Tensor which is backed by `this` Tensor.
         */
        VTensor<RANK> view();
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
        const VTensor<1> flattened() const;
        /**
         * Returns a View Tensor with rank 1 of `this` Tensor, all of it's values are
         * flattened into one vector.
         * The returned View Tensor is backed by `this` Tensor.
         */
        VTensor<1> flattened();

        /* Binary Operations */
        public:

        /**
         * Adds all of the elements in the `other` Tensor to all of the elements in
         * `this` Tensor and returns the result.
         * Both Tensors must be the same total size, but do not necessarily have to have
         * the same dimensions.
         * The dimensions of `this` Tensor are passed onto the result Tensor.
         */
        Tensor<RANK> operator + ( const BaseTensor<RANK>& other ) const;
        /**
         * Subtracts all of the elements in the `other` Tensor from all of the elements
         * in `this` Tensor and returns the result.
         * Both Tensors must be the same total size, but do not necessarily have to have
         * the same dimensions.
         * The dimensions of `this` Tensor are passed onto the result Tensor.
         */
        Tensor<RANK> operator - ( const BaseTensor<RANK>& other ) const;
        /**
         * Multiplies all of the elements in the `other` Tensor with all of the elements
         * in `this` Tensor and returns the result.
         * Both Tensors must be the same total size, but do not necessarily have to have
         * the same dimensions.
         * The dimensions of `this` Tensor are passed onto the result Tensor.
         */
        Tensor<RANK> operator * ( const BaseTensor<RANK>& other ) const;
        /**
         * Divides all of the elements in the `other` Tensor from all of the elements
         * in `this` Tensor and returns the result.
         * Both Tensors must be the same total size, but do not necessarily have to have
         * the same dimensions.
         * The dimensions of `this` Tensor are passed onto the result Tensor.
         */
        Tensor<RANK> operator / ( const BaseTensor<RANK>& other ) const;
        /**
         * Multiplies all of the elements in `this` Tensor by `scale` and returns the result.
        */
        template<size_t R>
        friend Tensor<R> operator * ( const BaseTensor<R>& tensor, float scale );
        /**
         * Multiplies all of the elements in `this` Tensor by `scale` and returns the result.
        */
        template<size_t R>
        friend Tensor<R> operator * ( float scale, const BaseTensor<R>& tensor );
        /**
         * Divides all of the elements in `this` Tensor by `scale` and returns the result.
        */
        Tensor<RANK> operator / ( float scale ) const;
        /**
         * Negates all of the elements in `this` Tensor and returns the result.
         */
        Tensor<RANK> operator - () const;
        /**
         * Returns a copy of `this` Tensor with the same dimensions, but with no data set.
         */
        Tensor<RANK> emptied() const;

        /**
         * Returns true if the `other` Tensor has the same dimensions as `this` Tensor, and false otherwise.
         */
        bool isSameSize( const BaseTensor<RANK>& other ) const;
        /**
         * Returns true if the `other` Tensor is equal to `this` Tensor, and false otherwise.
         */
        bool operator == ( const BaseTensor<RANK>& other ) const;
        /**
         * Returns true if the `other` Tensor is not equal to `this` Tensor, and false otherwise.
         */
        bool operator != ( const BaseTensor<RANK>& other ) const;

        /* General mutators */
        public:

        /** 
         * Adds all of the elements in the other Tensor to all of the elements in `this`
         * Tensor.
         * The `other` Tensor must be the same total size as `this` Tensor, but does not
         * necessarily have to have the same dimensions.
         */
        BaseTensor<RANK>& operator += ( const BaseTensor<RANK>& other );
        /**
         * Subtracts all of the elements in the other Tensor from all of the elements in
         * `this` Tensor.
         * The `other` Tensor must be the same total size as `this` Tensor, but does not 
         * necessarily have to have the same dimensions.
         */
        BaseTensor<RANK>& operator -= ( const BaseTensor<RANK>& other );
        /** 
         * Multiplies all of the elements in the `other` Tensor with all of the elements
         * in `this` Tensor.
         * The `other` Tensor must be the same total size as `this` Tensor, but does not
         * necessarily have to have the same dimensions.
         */
        BaseTensor<RANK>& operator *= ( const BaseTensor<RANK>& other );
        /**
         * Divides all of the elements in the `other` Tensor from all of the elements in
         * `this` Tensor.
         * The `other` Tensor must be the same total size as `this` Tensor, but does not 
         * necessarily have to have the same dimensions.
         */
        BaseTensor<RANK>& operator /= ( const BaseTensor<RANK>& other );
        /** 
         * Multiples all of the elements in `this` Tensor with the given `scale`.
         */
        BaseTensor<RANK>& operator *= ( float scale );
        /** 
         * Divides all of the elements in `this` Tensor with the given `scale`.
         */
        BaseTensor<RANK>& operator /= ( float scale );
        /**
         * This sets every value in `this` Tensor to `fill`.
         */
        BaseTensor<RANK>& fill( const float fill );
        /**
         * This sets the values in `this` Tensor to the values in `tensor`.
         * The given `tensor` must have the same dimensions as `this` Tensor.
         */
        BaseTensor<RANK>& set( const BaseTensor<RANK>& tensor );
        /**
         * This transforms each element in `this` Tensor using the given 
         * `transform_function`, which should return a `float`.
         * `transform_function` can have an argument of type `float` and/or an argument 
         * for the index (a `size_t` when RANK=1, and `size[RANK] when RANK>1`). The
         * arguments can be one or the other, and in any order.
         * The value in the Tensor is set to the returned value of `transform_function`.
         */
        template<typename Func>
        BaseTensor<RANK>& transform( Func transform_function );

        /* Vector operations */
        public:

        /**
         * Finds the magnitude of this Vector and returns the result.
         */
        float mag() const
        requires (RANK == 1);
        /**
         * Finds the squared magnitude of this Vector and returns the result.
         */
        float squaredMag() const
        requires (RANK == 1);
        /**
         * Normalizes `this` Vector, and returns the result.
         */
        Tensor<1> normalized() const
        requires (RANK == 1);
        /**
         * Takes the dot product of this Vector with the other Vector and returns the result.
         * The two vectors must be the same size.
         */
        float dot( const BaseTensor<1>& other ) const
        requires (RANK == 1);
        /**
         * Takes the cross product of this Vector with the other Vector and returns the result.
         * The two vectors must have a size of 3.
         */
        Tensor<1> cross( const BaseTensor<1>& other ) const
        requires (RANK == 1);

        /**
         * Finds the average of the elements in this Vector and returns the result.
         */
        float average() const
        requires (RANK == 1);

        /* Matrix operations */
        public:

        /**
         * Takes the transpose of `this` Matrix and returns the result.
         * If `this` Matrix is of size (m x n), then the result will be of size (n x m).
         */
        Tensor<2> transpose() const
        requires (RANK == 2);
        /**
         * Takes the transpose of `this` Vector and returns the result.
         * If `this` Vector of is size (m), then the result will be of size (1 x m).
         */
        Tensor<2> transpose() const
        requires (RANK == 1);
        /**
         * Finds the matrix multiplication of the `other` Matrix on `this` Matrix and
         * returns the result.
         * `this` Matrix must be of size (m x n) and the `other` Matrix must be of size
         * (n x w)
         */
        Tensor<2> mul( const BaseTensor<2>& other ) const
        requires (RANK == 2);
        /**
         * Finds the matrix multiplication of the `other` Vector on `this` Matrix and
         * returns the result.
         * `this` matrix must be of size (m x n) and the `other` Vector must be of size
         * (n).
         */
        Tensor<1> mul( const BaseTensor<1>& other ) const
        requires (RANK == 2);
        /**
         * Finds the matrix multiplication of the `other` Matrix on `this` Vector and 
         * returns the result.
         * `this` vector must be of size (m) and the `other` matrix must be of size
         * (1 x n).
         */
        Tensor<2> mul( const BaseTensor<2>& other ) const
        requires (RANK == 1);
        /**
         * Finds the determinant of `this` Matrix and returns the result. 
         * `this` Matrix must be of size (n x n).
         */
        float determinant() const
        requires (RANK == 2);
        /**
         * Finds the matrix inverse of `this` Matrix and returns the result. 
         * `this` Matrix must be of size (n x n) and invertible (the columns are linearly
         * independent).
         */
        Tensor<2> inverse() const
        requires (RANK == 2);

        /* Getters */
        public:

        /**
         * Returns the rank of the tensor (the number of dimensions).
         * NOTE: This is NOT the same as the matrix rank.
         */
        size_t rank() const;
        /**
         * Returns the total size of the Tensor (the total number of elements).
         */
        size_t totalSize() const;
        /**
         * Returns a pointer to the start of the contiguous data stored in the Tensor.
         */
        const float* data() const;
        /**
         * Defined for RANK=1 Tensors, this returns the size of the Tensor.
         * This is the same as calling totalSize()
         */
        size_t size() const
        requires (RANK == 1);
        /**
         * Defined for RANK>1 Tensors, this returns the size of the given dimension.
         */
        size_t size( size_t dimension ) const;

        /**
         * Prints out the Tensor as a string.
         */
        template<size_t R>
        friend std::ostream& operator << ( std::ostream& fs, const BaseTensor<R>& t );

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
        float* data_;
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
        template<size_t R>
        friend class BaseTensor;
        template<size_t R>
        friend class Tensor;
        template<size_t R>
        friend class VTensor;
        /**
         * Declare RaggedTensor as a friend of BaseTensor so that it can view and manage
         * internal Tensor and VTensor data.
         */
        template<size_t R>
        friend class RaggedTensor;
    };


    /**
     * This represents a Tensor itself, which contains and manages all of it's own data.
     * Any instance of a Tensor, and the data contained within, is managed by itself.
     */
    template<size_t RANK>
    class Tensor : public BaseTensor<RANK> {
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
        Tensor( size_t dim, float fill )
        requires (RANK == 1);
        /**
         * Defined for RANK=1 Tensors, constructs a Tensor with the given dimensions and
         * set with the values from `fill`. `fill` must be have valid memory from index 0
         * to `dim-1`.
         * Throws an error if `dim` is equal to 0.
         */
        Tensor( size_t dim, const float fill[] )
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
        Tensor( const size_t (&dims)[RANK], float fill );
        /**
         * Constructs a Tensor initialized with the given `elements`.
         * Throws an error if `elements` or any inner elements inside `elements` has a
         * size of 0.
         * Throws an error if the `elements` are non-rectangular.
         */
        Tensor( InitializerElements<RANK> elements );
        /**
         * Defined for RANK>1 Tensors, constructs a Tensor initialized with the given
         * `Tensor<RANK-1>` elements. The size of the first dimension is the size of
         * `elements`.
         * Throws an error if `elements` has a size of 0.
         * Throws an error if any of the Tensors in `elements` have differing dimensions.
         */
        Tensor( std::initializer_list<std::reference_wrapper<const BaseTensor<RANK-1>>> elements )
        requires (RANK > 1);

        /**
         * Copy constructor.
         */
        Tensor( const Tensor<RANK>& other );
        /**
         * Copy constructor from BaseTensor.
         */
        Tensor( const BaseTensor<RANK>& other );
        /**
         * Move constructor.
         */
        Tensor( Tensor<RANK>&& other );
        /**
         * Destructor.
         */
        ~Tensor();
        /**
         * Assignment operator.
         * Ensures that memory is freed when existing object is overwritten.
         * Any VTensors referring to `this` Tensor will be invalidated.
         */
        Tensor<RANK>& operator = ( const Tensor<RANK>& other );
        /**
         * Assignment operator from BaseTensor.
         * Ensures that memory is freed when existing object is overwritten.
         * Any VTensors referring to `this` Tensor will be invalidated.
         */
        Tensor<RANK>& operator = ( const BaseTensor<RANK>& other );
        /**
         * Move assignment operator.
         * Ensures that memory is freed when existing object is overwritten.
         * Any VTensors referring to `this` Tensor will be invalidated.
         */
        Tensor<RANK>& operator = ( Tensor<RANK>&& other );
    
        /* Factory functions */
        public:

        /**
         * Creates a square identity (`dims` x `dims`) matrix with the given
         * `diagonal_value`.
         */
        static Tensor<2> identity( const size_t dims, const float diagonal_value = 1.0f );
    };


    /**
     * This represents the view into a part or whole of a Tensor.
     * Any instance of a VTensor, and the data contained within, is backed by a Tensor.
     * Despite its name, a VTensor can be modified, but it will also modify the Tensor it is backed by.
     */
    template<size_t RANK>
    class VTensor : public BaseTensor<RANK> {
        /* Constructors */
        public:

        /**
         * Constructs an empty VTensor with a size of 0 in each dimension.
         */
        VTensor();
        /**
         * Copy constructor.
         */
        VTensor( const VTensor<RANK>& other );
        /**
         * Assignment operator.
         */
        VTensor<RANK>& operator = ( const VTensor<RANK>& other );
    };


    /**
     * This represents a tensor whose first dimensions elements do not have the same
     * size. Each inner tensor has a rank of RANK-1, but can have differing sizes.
     * NOTE: Name stolen from PyTorch, though unsure if functionally similar.
     */
    template<size_t RANK>
    class RaggedTensor {
        // Ensure that Ragged Tensor RANK cannot be less than 1 (must have 2 or more dimensions)
        static_assert(RANK > 1, "Ragged Tensor rank cannot be less than 1.");

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
         * Constructs a RaggedTensor containing the tensors specified in `elements`.
         */
        RaggedTensor( std::initializer_list<std::reference_wrapper<const BaseTensor<RANK-1>>> elements );

        /**
         * Copy constructor.
         */
        RaggedTensor( const RaggedTensor<RANK>& other );
        /**
         * Copy constructor from BaseTensor.
         */
        RaggedTensor( const BaseTensor<RANK>& other );
        /**
         * Move constructor.
         */
        RaggedTensor( RaggedTensor<RANK>&& other );
        /**
         * Destructor.
         */
        ~RaggedTensor();
        /**
         * Assignment operator.
         * Ensures that memory is freed when existing object is overwritten.
         */
        RaggedTensor<RANK>& operator = ( const RaggedTensor<RANK>& other );
        /**
         * Assignment operator from BaseTensor.
         * Ensures that memory is freed when existing object is overwritten.
         */
        RaggedTensor<RANK>& operator = ( const BaseTensor<RANK>& other );
        /**
         * Move assignment operator.
         * Ensures that memory is freed when existing object is overwritten.
         */
        RaggedTensor<RANK>& operator = ( RaggedTensor<RANK>&& other );

        /* Accessors */
        public:

        /**
         * Returns the element at the given `indexes`.
         */
        const float& operator [] ( const size_t (&indexes)[RANK] ) const;
        /**
         * Returns a mutable reference to the element at the given `indexes`.
         */
        float& operator [] ( const size_t (&indexes)[RANK] );
        /**
         * This returns the inner Tensor at the given index in the first dimension.
         */
        const VTensor<RANK-1> operator [] ( size_t index ) const;
        /**
         * This returns a mutable reference to the inner Tensor at the given index in the
         * first dimension.
         */
        VTensor<RANK-1> operator [] ( size_t index );

        /* Binary Operations */
        public:

        /**
         * Adds all of the elements in the `other` RaggedTensor to all of the elements in
         * `this` RaggedTensor and returns the result.
         * Both RaggedTensors must be the same total size, but do not necessarily have to
         * have the same dimensions.
         * The dimensions of `this` RaggedTensor are passed onto the result RaggedTensor.
         */
        RaggedTensor<RANK> operator + ( const RaggedTensor<RANK>& other ) const;
        /**
         * Subtracts all of the elements in the `other` RaggedTensor from all of the
         * elements in `this` RaggedTensor and returns the result.
         * Both RaggedTensors must be the same total size, but do not necessarily have to
         * have the same dimensions.
         * The dimensions of `this` RaggedTensor are passed onto the result RaggedTensor.
         */
        RaggedTensor<RANK> operator - ( const RaggedTensor<RANK>& other ) const;
        /**
         * Multiplies all of the elements in the `other` RaggedTensor with all of the
         * elements in `this` RaggedTensor and returns the result.
         * Both RaggedTensors must be the same total size, but do not necessarily have to
         * have the same dimensions.
         * The dimensions of `this` RaggedTensor are passed onto the result RaggedTensor.
         */
        RaggedTensor<RANK> operator * ( const RaggedTensor<RANK>& other ) const;
        /**
         * Divides all of the elements in the `other` RaggedTensor from all of the
         * elements in `this` RaggedTensor and returns the result.
         * Both RaggedTensors must be the same total size, but do not necessarily have to
         * have the same dimensions.
         * The dimensions of `this` RaggedTensor are passed onto the result RaggedTensor.
         */
        RaggedTensor<RANK> operator / ( const RaggedTensor<RANK>& other ) const;
        /**
         * Multiplies all of the elements in `this` RaggedTensor by `scale` and returns
         * the result.
        */
        template<size_t R>
        friend RaggedTensor<R> operator * ( const RaggedTensor<R>& tensor, float scale );
        /**
         * Multiplies all of the elements in `this` RaggedTensor by `scale` and returns
         * the result.
        */
        template<size_t R>
        friend RaggedTensor<R> operator * ( float scale, const RaggedTensor<R>& tensor );
        /**
         * Divides all of the elements in `this` RaggedTensor by `scale` and returns the
         * result.
        */
        RaggedTensor<RANK> operator / ( float scale ) const;
        /**
         * Negates all of the elements in `this` RaggedTensor and returns the result.
         */
        RaggedTensor<RANK> operator - () const;
        /**
         * Returns a copy of `this` RaggedTensor with the same dimensions, but with no
         * data set.
         */
        RaggedTensor<RANK> emptied() const;

        /**
         * Returns true if the `other` RaggedTensor has the same dimensions as `this` RaggedTensor, and false otherwise.
         */
        bool isSameSize( const RaggedTensor<RANK>& other ) const;
        /**
         * Returns true if the `other` RaggedTensor is equal to `this` RaggedTensor, and false otherwise.
         */
        bool operator == ( const RaggedTensor<RANK>& other ) const;
        /**
         * Returns true if the `other` RaggedTensor is not equal to `this` RaggedTensor, and false otherwise.
         */
        bool operator != ( const RaggedTensor<RANK>& other ) const;
        
        /* General mutators */
        public:

        /** 
         * Adds all of the elements in the other RaggedTensor to all of the elements in
         * `this` RaggedTensor.
         * The `other` RaggedTensor must be the same total size as `this` RaggedTensor,
         * but does not necessarily have to have the same dimensions.
         */
        RaggedTensor<RANK>& operator += ( const RaggedTensor<RANK>& other );
        /**
         * Subtracts all of the elements in the other RaggedTensor from all of the
         * elements in `this` RaggedTensor.
         * The `other` RaggedTensor must be the same total size as `this` RaggedTensor,
         * but does not necessarily have to have the same dimensions.
         */
        RaggedTensor<RANK>& operator -= ( const RaggedTensor<RANK>& other );
        /** 
         * Multiplies all of the elements in the `other` RaggedTensor to all of the
         * elements in `this` RaggedTensor.
         * The `other` RaggedTensor must be the same total size as `this` RaggedTensor,
         * but does not necessarily have to have the same dimensions.
         */
        RaggedTensor<RANK>& operator *= ( const RaggedTensor<RANK>& other );
        /**
         * Divides all of the elements in the `other` RaggedTensor from all of the
         * elements in `this` RaggedTensor.
         * The `other` RaggedTensor must be the same total size as `this` RaggedTensor,
         * but does not necessarily have to have the same dimensions.
         */
        RaggedTensor<RANK>& operator /= ( const RaggedTensor<RANK>& other );
        /** 
         * Multiples all of the elements in `this` RaggedTensor with the given `scale`.
         */
        RaggedTensor<RANK>& operator *= ( float scale );
        /** 
         * Divides all of the elements in `this` RaggedTensor by the given `scale`.
         */
        RaggedTensor<RANK>& operator /= ( float scale );
        /**
         * This sets every value in `this` RaggedTensor to `fill`.
         */
        RaggedTensor<RANK>& fill( const float fill );
        /**
         * This sets the values in `this` RaggedTensor to the values in `tensor`.
         * The given `tensor` must have the same dimensions as `this` RaggedTensor.
         */
        RaggedTensor<RANK>& set( const RaggedTensor<RANK>& tensor );
        /**
         * This transforms each element in `this` RaggedTensor using the given 
         * `transform_function`. The only argument the function should take is of type
         * `float`, and the function should return a `float`.
         * The value in the RaggedTensor is set to the returned value of
         * `transform_function`.
         */
        template<typename Func>
        RaggedTensor<RANK>& transform( Func transform_function );

        /* Getters */
        public:

        /**
         * Returns the rank of the tensor (the number of dimensions).
         * NOTE: This is NOT the same as the matrix rank.
         */
        size_t rank() const;
        /**
         * Returns the total size of the RaggedTensor (the total number of elements).
         */
        size_t totalSize() const;
        /**
         * The pointer to the allocated data in this RaggedTensor.
         */
        const float* data() const;
        /**
         * Returns the size of the first dimension of this RaggedTensor.
         */
        size_t dim1Size() const;

        /* Member Variables */
        protected:

        /**
         * The total number of elements in the RaggedTensor.
         * This is the sum of the total size of each inner Tensor.
         */
        size_t total_size;
        /**
         * The pointer to the allocated data in this Tensor.
         * The memory from `data` to `data + total_size - 1` will always be valid.
         */
        float* data_;
        /**
         * The size of the first dimension. More simply, the number of inner tensors
         */
        size_t dimension1;
        /**
         * The View Tensors which keep track of the size and memory locations of the
         * inner tensors.
         */
        VTensor<RANK-1>* inner_tensors;
    };


    /* Vector and Matrix type definitions */
    using BaseVector = BaseTensor<1>;
    using Vector = Tensor<1>;
    using VVector = VTensor<1>;
    using BaseMatrix = BaseTensor<2>;
    using Matrix = Tensor<2>;
    using VMatrix = VTensor<2>;
    using RaggedMatrix = RaggedTensor<2>;
}



/* Implementation */
namespace jai {
    /* Implementation Helper Functions For Bulk Operations */

    namespace {
        inline void fillValues( const float A, float* dest, const size_t size ) {
            std::fill(dest, dest + size, A);
        }
        inline void setValues( const float* src_A, float* dest, const size_t size ) {
            std::memcpy( dest, src_A, size * sizeof(float) );
        }
        inline void addValues( const float* src_A, const float* src_B, float* dest, const size_t size ) {
            for( size_t i = 0; i < size; ++i ) {
                dest[i] = src_A[i] + src_B[i];
            }
        }
        inline void subtractValues( const float* src_A, const float* src_B, float* dest, const size_t size ) {
            for( size_t i = 0; i < size; ++i ) {
                dest[i] = src_A[i] - src_B[i];
            }
        }
        inline void multiplyValues( const float* src_A, const float* src_B, float* dest, const size_t size ) {
            for( size_t i = 0; i < size; ++i ) {
                dest[i] = src_A[i] * src_B[i];
            }
        }
        inline void divideValues( const float* src_A, const float* src_B, float* dest, const size_t size ) {
            for( size_t i = 0; i < size; ++i ) {
                dest[i] = src_A[i] / src_B[i];
            }
        }
        inline void multiplyValuesByScalar( const float* src_A, const float B, float* dest, const size_t size ) {
            for( size_t i = 0; i < size; ++i ) {
                dest[i] = src_A[i] * B;
            }
        }
        inline void negateValues( const float* src_A, float* dest, const size_t size ) {
            for( size_t i = 0; i < size; ++i ) {
                dest[i] = -src_A[i];
            }
        }
        inline bool compareValuesForEquality( const float* src_A, const float* src_B, const size_t size ) {
            for( size_t i = 0; i < size; ++i ) {
                if( src_A[i] != src_B[i] ) {
                    return false;
                }
            }
            return true;
        }
    }


    /* BaseTensor Implementation */

    template<size_t RANK>
    BaseTensor<RANK>::BaseTensor() { }

    template<size_t RANK>
    const float& BaseTensor<RANK>::operator [] ( const size_t index ) const 
    requires (RANK == 1) {
        return this->data_[index];
    }
    
    template<size_t RANK>
    float& BaseTensor<RANK>::operator [] ( const size_t index ) 
    requires (RANK == 1) {
        return this->data_[index];
    }
    
    template<size_t RANK>
    const float& BaseTensor<RANK>::operator [] ( const size_t (&indexes)[RANK] ) const {
        size_t index = 0;
        size_t inner_tensor_size = 1;
        for( size_t i = 0; i < RANK; ++i ) {
            index += inner_tensor_size * indexes[RANK - i - 1];
            inner_tensor_size *= this->dimensions[RANK - i - 1];
        }
        return this->data_[index];
    }
    
    template<size_t RANK>
    float& BaseTensor<RANK>::operator [] ( const size_t (&indexes)[RANK] ) {
        size_t index = 0;
        size_t inner_tensor_size = 1;
        for( size_t i = 0; i < RANK; ++i ) {
            index += inner_tensor_size * indexes[RANK - i - 1];
            inner_tensor_size *= this->dimensions[RANK - i - 1];
        }
        return this->data_[index];
    }
    
    template<size_t RANK>
    const VTensor<RANK-1> BaseTensor<RANK>::operator [] ( const size_t index ) const 
    requires (RANK > 1) {
        VTensor<RANK-1> inner_view;
        for( size_t i = 0; i < RANK-1; ++i ) {
            inner_view.dimensions[i] = this->dimensions[i+1];
        }
        const size_t inner_tensor_total_size = this->total_size / this->dimensions[0];
        inner_view.total_size = inner_tensor_total_size;
        inner_view.data_ = this->data_ + inner_tensor_total_size*index;
        return inner_view;
    }
    
    template<size_t RANK>
    VTensor<RANK-1> BaseTensor<RANK>::operator [] ( const size_t index ) 
    requires (RANK > 1) {
        VTensor<RANK-1> inner_view;
        for( size_t i = 0; i < RANK-1; ++i ) {
            inner_view.dimensions[i] = this->dimensions[i+1];
        }
        const size_t inner_tensor_total_size = this->total_size / this->dimensions[0];
        inner_view.total_size = inner_tensor_total_size;
        inner_view.data_ = this->data_ + inner_tensor_total_size*index;
        return inner_view;
    }
 
    template<size_t RANK>
    const VTensor<RANK> BaseTensor<RANK>::view() const {
        VTensor<RANK> view;
        for( size_t i = 0; i < RANK; ++i ) {
            view.dimensions[i] = this->dimensions[i];
        }
        view.total_size = this->total_size;
        view.data_ = this->data_;
        return view;
    }
    
    template<size_t RANK>
    VTensor<RANK> BaseTensor<RANK>::view() {
        VTensor<RANK> view;
        for( size_t i = 0; i < RANK; ++i ) {
            view.dimensions[i] = this->dimensions[i];
        }
        view.total_size = this->total_size;
        view.data_ = this->data_;
        return view;
    }
    
    template<size_t RANK>
    const VTensor<RANK+1> BaseTensor<RANK>::rankUp() const {
        VTensor<RANK+1> ranked_up_view;
        for( size_t i = 0; i < RANK; ++i ) {
            ranked_up_view.dimensions[i] = this->dimensions[i];
        }
        ranked_up_view.dimensions[RANK] = 1;
        ranked_up_view.total_size = this->total_size;
        ranked_up_view.data_ = this->data_;
        return ranked_up_view;
    }
    
    template<size_t RANK>
    VTensor<RANK+1> BaseTensor<RANK>::rankUp() {
        VTensor<RANK+1> ranked_up_view;
        for( size_t i = 0; i < RANK; ++i ) {
            ranked_up_view.dimensions[i] = this->dimensions[i];
        }
        ranked_up_view.dimensions[RANK] = 1;
        ranked_up_view.total_size = this->total_size;
        ranked_up_view.data_ = this->data_;
    }
    
    template<size_t RANK>
    const VTensor<1> BaseTensor<RANK>::flattened() const {
        VTensor<1> flattened_view;
        flattened_view.dimensions[0] = this->total_size;
        flattened_view.total_size = this->total_size;
        flattened_view.data_ = this->data_;
        return flattened_view;
    }
    
    template<size_t RANK>
    VTensor<1> BaseTensor<RANK>::flattened() {
        VTensor<1> flattened_view;
        flattened_view.dimensions[0] = this->total_size;
        flattened_view.total_size = this->total_size;
        flattened_view.data_ = this->data_;
        return flattened_view;
    }

    template<size_t RANK>
    Tensor<RANK> BaseTensor<RANK>::operator + ( const BaseTensor<RANK>& other ) const {
        Tensor<RANK> result = this->emptied();
        // Add the values in the Tensors together
        addValues(this->data_, other.data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK>
    Tensor<RANK> BaseTensor<RANK>::operator - ( const BaseTensor<RANK>& other ) const {
        Tensor<RANK> result = this->emptied();
        // Subtract the values in this Tensor by the values in the other Tensor
        subtractValues(this->data_, other.data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK>
    Tensor<RANK> BaseTensor<RANK>::operator * ( const BaseTensor<RANK>& other ) const {
        Tensor<RANK> result = this->emptied();
        // Multiply the values in the Tensors together
        multiplyValues(this->data_, other.data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK>
    Tensor<RANK> BaseTensor<RANK>::operator / ( const BaseTensor<RANK>& other ) const {
        Tensor<RANK> result = this->emptied();
        // Divide the values in this Tensor by the values in the other Tensor
        divideValues(this->data_, other.data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK>
    Tensor<RANK> operator * ( const BaseTensor<RANK>& tensor, const float scale ) {
        Tensor<RANK> result = tensor.emptied();
        // Multiply the values in the Tensor by the scale
        multiplyValuesByScalar(tensor.data_, scale, result.data_, tensor.total_size);

        return result;
    }
    
    template<size_t RANK>
    Tensor<RANK> operator * ( const float scale, const BaseTensor<RANK>& tensor ) {
        Tensor<RANK> result = tensor.emptied();
        // Multiply the values in the Tensor by the scale
        multiplyValuesByScalar(tensor.data_, scale, result.data_, tensor.total_size);
        
        return result;
    }
    
    template<size_t RANK>
    Tensor<RANK> BaseTensor<RANK>::operator / ( const float scale ) const {
        Tensor<RANK> result = this->emptied();
        // Multiply the values in this Tensor by the inverted scale
        multiplyValuesByScalar(this->data_, (1.0f / scale), result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK>
    Tensor<RANK> BaseTensor<RANK>::operator - () const {
        Tensor<RANK> result = this->emptied();
        // Negate the values in this Tensor
        negateValues(this->data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK>
    Tensor<RANK> BaseTensor<RANK>::emptied() const {
        return Tensor<RANK>(this->dimensions);
    } 

    template<size_t RANK>
    bool BaseTensor<RANK>::isSameSize( const BaseTensor<RANK>& other ) const {
        for( size_t i = 0; i < RANK; ++i ) {
            if( this->dimensions[i] != other.dimensions[i] ) {
                return false;
            }
        }
        return true;
    }
    
    template<size_t RANK>
    bool BaseTensor<RANK>::operator == ( const BaseTensor<RANK>& other ) const {
        if( !this->isSameSize(other) ) {
            return false;
        }

        return compareValuesForEquality(this->data_, other.data_, this->total_size);
    }
    
    template<size_t RANK>
    bool BaseTensor<RANK>::operator != ( const BaseTensor<RANK>& other ) const {
        return !(*this == other);
    }

    template<size_t RANK>
    BaseTensor<RANK>& BaseTensor<RANK>::operator += ( const BaseTensor<RANK>& other ) {
        // Add other's values
        addValues(this->data_, other.data_, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK>
    BaseTensor<RANK>& BaseTensor<RANK>::operator -= ( const BaseTensor<RANK>& other ) {
        // Subtract other's values
        subtractValues(this->data_, other.data_, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK>
    BaseTensor<RANK>& BaseTensor<RANK>::operator *= ( const BaseTensor<RANK>& other ) {
        // Multiply other's values
        multiplyValues(this->data_, other.data_, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK>
    BaseTensor<RANK>& BaseTensor<RANK>::operator /= ( const BaseTensor<RANK>& other ) {
        // Divide other's values
        divideValues(this->data_, other.data_, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK>
    BaseTensor<RANK>& BaseTensor<RANK>::operator *= ( const float scale ) {
        // Multiply by scale
        multiplyValuesByScalar(this->data_, scale, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK>
    BaseTensor<RANK>& BaseTensor<RANK>::operator /= ( const float scale ) {
        // Divide by scale
        multiplyValuesByScalar(this->data_, 1.0f / scale, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK>
    BaseTensor<RANK>& BaseTensor<RANK>::fill( const float fill ) {
        fillValues(fill, this->data, this->total_size);
        return *this;
    }
    
    template<size_t RANK>
    BaseTensor<RANK>& BaseTensor<RANK>::set( const BaseTensor<RANK>& tensor ) {
        setValues(tensor.data_, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK>
    
    template<typename Func>
    BaseTensor<RANK>& BaseTensor<RANK>::transform( Func transform_function ) {
        // Define a types for the value and index, so that the value is passed when
        // `float` is defined as the argument and so that the index is passed when 
        // `size_t` is defined as the argument.
        // Since `float` and `size_t` can be implicitly converted between each other
        // these types are needed to differentiate between a `float` and `size_t`
        // argument in `transform_function`.
        struct Value_t {
            operator float();
            operator size_t() = delete;
        };
        struct Index_t {
            operator size_t();
            operator float() = delete;
        };
        
        // If the function takes no arguments
        if constexpr( std::is_invocable_r_v<float, Func> ) {
            for( size_t i = 0; i < this->total_size; ++i ) {
                this->data_[i] = transform_function();
            }
            return *this;
        }

        // If the function just takes the values, but not any indexes
        else if constexpr( std::is_invocable_r_v<float, Func, Value_t> ) {
            for( size_t i = 0; i < this->total_size; ++i ) {
                this->data_[i] = transform_function(this->data_[i]);
            }
            return *this;
        }
        
        // For RANK=1 Tensors
        else if constexpr( RANK == 1 ) {
            for( size_t i = 0; i < this->total_size; ++i ) {
                // If the function just accepts an index
                if constexpr( std::is_invocable_r_v<float, Func, Index_t> ) {
                    this->data_[i] = transform_function(i);
                }

                // If the function accepts an index and then a tensor value
                else if constexpr( std::is_invocable_r_v<float, Func, Index_t, Value_t> ) {
                    this->data_[i] = transform_function(i, this->data_[i]);
                }

                // If the function accepts a tensor value and then an index
                else if constexpr( std::is_invocable_r_v<float, Func, Value_t, Index_t> ) { 
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
                if constexpr( std::is_invocable_r_v<float, Func, size_t[RANK]> ) {
                    (*this)[indexes] = transform_function(indexes);
                }

                // If the function accepts an index and then a tensor value
                else if constexpr( std::is_invocable_r_v<float, Func, size_t[RANK], float> ) {
                    (*this)[indexes] = transform_function(indexes, (*this)[indexes]);
                }

                // If the function accepts a tensor value and then an index
                else if constexpr( std::is_invocable_r_v<float, Func, float, size_t[RANK]> ) {
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

    template<size_t RANK>
    float BaseTensor<RANK>::mag() const 
    requires (RANK == 1) {
        return std::sqrt(this->squaredMag());
    }
    
    template<size_t RANK>
    float BaseTensor<RANK>::squaredMag() const 
    requires (RANK == 1) {
        float sqrd_sum = 0;
        for( size_t i = 0; i < this->total_size; ++i ) {
            sqrd_sum += this->data_[i] * this->data_[i];
        }
        return sqrd_sum;
    }
    
    template<size_t RANK>
    Tensor<1> BaseTensor<RANK>::normalized() const 
    requires (RANK == 1) {
        const float mag = this->mag();
        return (*this) / mag;
    }
    
    template<size_t RANK>
    float BaseTensor<RANK>::dot( const BaseTensor<1>& other ) const 
    requires (RANK == 1) {
        float sum = 0;
        for( size_t i = 0; i < this->total_size; ++i ) {
            sum += this->data_[i] * other.data_[i];
        }
        return sum;
    }
    
    template<size_t RANK>
    Tensor<1> BaseTensor<RANK>::cross( const BaseTensor<1>& other ) const 
    requires (RANK == 1) {
        Tensor<1> result(3);
        result[0] = this[1] * other[2] - this[2] * other[1];
        result[1] = this[2] * other[0] - this[0] * other[2];
        result[2] = this[0] * other[1] - this[1] * other[0];
        return result;
    }
    
    template<size_t RANK>
    float BaseTensor<RANK>::average() const
    requires (RANK == 1) {
        float sum = 0;
        for( size_t i = 0; i < this->total_size; ++i ) {
            sum += this->data_[i];
        }
        return sum / this->total_size;
    }

    template<size_t RANK>
    Tensor<2> BaseTensor<RANK>::transpose() const 
    requires (RANK == 2) {
        Tensor<2> result({this->dimensions[1], this->dimensions[0]});
        for( size_t i = 0; i < this->dimensions[0]; ++i ) {
            for( size_t j = 0; j < this->dimensions[1]; ++j ) {
                result[{j, i}] = (*this)[{i, j}];
            }
        }
        return result;
    }
    
    template<size_t RANK>
    Tensor<2> BaseTensor<RANK>::transpose() const 
    requires (RANK == 1) {
        Tensor<2> result({0, this->dimensions[0]});
        setValues(this->data_, result.data_, this->total_size);
        return result;
    }
    
    template<size_t RANK>
    Tensor<2> BaseTensor<RANK>::mul( const BaseTensor<2>& other ) const 
    requires (RANK == 2) {
        // Create result Tensor
        Tensor<2> result({this->dimensions[0], other.dimensions[1]});
        // Perform matrix multiplication
        for( size_t i = 0; i < result.dimensions[0]; ++i ) {
            for( size_t j = 0; j < result.dimensions[1]; ++j ) {
                float sum = 0;
                for( size_t k = 0; k < this->dimensions[1]; ++k ) {
                    sum += (*this)[{i, k}] * other[{k, j}];
                }
                result[{i, j}] = sum;
            }
        }
        return result;
    }
    
    template<size_t RANK>
    Tensor<1> BaseTensor<RANK>::mul( const BaseTensor<1>& other ) const 
    requires (RANK == 2) {
        // Create result Tensor
        Tensor<1> result(this->dimensions[0]);
        // Perform matrix multiplication
        for( size_t i = 0; i < result.dimensions[0]; ++i ) {
            float sum = 0;
            for( size_t j = 0; j < result.dimensions[1]; ++j ) {
                sum += (*this)[{j, i}] * other[i];
            }
            result[i] = sum;
        }
        return result;
    }
    
    template<size_t RANK>
    Tensor<2> BaseTensor<RANK>::mul( const BaseTensor<2>& other ) const 
    requires (RANK == 1) {
        // Create result Tensor
        Tensor<2> result(this->dimensions[0], other.dimensions[1]);
        // Perform matrix multiplication
        for( size_t i = 0; i < result.dimensions[0]; ++i ) {
            for( size_t j = 0; j < result.dimensions[1]; ++j ) {
                result[{i, j}] = (*this)[i] * other[{0, j}];
            }
        }
        return result;
    }
    
    template<size_t RANK>
    float BaseTensor<RANK>::determinant() const 
    requires (RANK == 2) {
        const size_t size = this->dimensions[0];
        if( size == 2 ) {
            return this->data_[0] * this->data_[3] - this->data_[1] * this->data_[2];
        }

        float determinant = 0;
        float sign = -1;
        for( size_t k = 0; k < size; ++k ) {
            sign *= -1;

            // Get coefficient
            const float value_0_i = (*this)[{0, k}];
            // If the value is 0, skip finding determinant
            if( value_0_i == 0.0f ) {
                continue;
            }

            // Create sub matrix
            Tensor<2> sub_matrix({size - 1, size - 1});
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
    
    template<size_t RANK>
    Tensor<2> BaseTensor<RANK>::inverse() const 
    requires (RANK == 2) {
        Tensor<2> this_copy = this->transpose();
        Tensor<2> result = Tensor<2>::identity(this->dimensions[0]);

        const size_t size = this->dimensions[0];
        for( size_t i = 0; i < size; ++i ) {
            // Divide ith row by ith value in row
            const float diagonal_i = this_copy[{i, i}];
            if ( diagonal_i == 0 ) {
                break;
            }
            const float scale_i = 1.0f / diagonal_i;
            this_copy[i] *= scale_i;
            result[i] *= scale_i;

            // Delete values in ith column below this row
            for( size_t j = i + 1; j < size; ++j ) {
                const float sub_row_scale = this_copy[{j, i}];
                this_copy[j] -= this_copy[i] * sub_row_scale;
                result[j] -= result[i] * sub_row_scale;
            }

            // Delete values in ith column above this row
            for( size_t j = 0; j < i; ++j ) {
                const Tensor<1> row_sub = this_copy[i] * this_copy[{j, i}];
                const float sub_row_scale = this_copy[{j, i}];
                this_copy[j] -= this_copy[i] * sub_row_scale;
                result[j] -= result[i] * sub_row_scale;
            }
        }

        return result.transpose();
    }

    template<size_t RANK>
    size_t BaseTensor<RANK>::rank() const {
        return RANK;
    }
    
    template<size_t RANK>
    size_t BaseTensor<RANK>::totalSize() const {
        return this->total_size;
    }
    
    template<size_t RANK>
    const float* BaseTensor<RANK>::data() const {
        return this->data_;
    }
    
    template<size_t RANK>
    size_t BaseTensor<RANK>::size() const 
    requires (RANK == 1) {
        return this->total_size;
    }
    
    template<size_t RANK>
    size_t BaseTensor<RANK>::size( const size_t dimension ) const {
        return this->dimensions[dimension];
    }

    template<size_t RANK>
    std::ostream& operator << ( std::ostream& fs, const BaseTensor<RANK>& t ) {
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
        
        template<size_t RANK>
        size_t countInitializerElements( const InitializerElements<RANK>& elements, size_t dims[RANK] ) {
            dims[0] = elements.size();

            // RANK=1 case
            if constexpr( RANK == 1 ) {
                return elements.size();
            }

            // RANK>1 case
            else {
                const size_t inner_size = countInitializerElements<RANK-1>(*elements.begin(), dims + 1);
                return elements.size() * inner_size;
            }
        }
        
        template<size_t RANK>
        bool checkInitializerElements( const InitializerElements<RANK>& elements, const size_t dims[RANK] ) {
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
                    if( !checkInitializerElements<RANK-1>(inner_elements, dims + 1) ) {
                        return false;
                    }
                }
                return true;
            }
        }
        
        template<size_t RANK>
        void flattenInitializerElements( const InitializerElements<RANK>& elements, float*& data ) {
            // RANK=1 case
            if constexpr( RANK == 1 ) {
                for( const float element : elements ) {
                    *data = element;
                    ++data;
                }
            }

            // RANK>1 case
            else {
                for( const auto& inner_elements : elements ) {
                    flattenInitializerElements<RANK-1>(inner_elements, data);
                }
            }
        }
    }
    

    /* Tensor Implementation */

    template<size_t RANK>
    Tensor<RANK>::Tensor() {
        // Set all dimensions to 0
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = 0;
        }
        // Allocate no memory
        this->total_size = 0;
        this->data_ = nullptr;
    }
    
    template<size_t RANK>
    Tensor<RANK>::Tensor( const size_t dim ) 
    requires (RANK == 1) {
        if( dim == 0 ) {
            throw std::invalid_argument("The dimension size is less than 1.");
        }
        this->dimensions[0] = dim;
        this->total_size = dim;
        this->data_ = new float[dim];
    }
    
    template<size_t RANK>
    Tensor<RANK>::Tensor( const size_t dim, const float fill ) 
    requires (RANK == 1) {
        if( dim == 0 ) {
            throw std::invalid_argument("The dimension size is less than 1.");
        }
        this->dimensions[0] = dim;
        this->total_size = dim;
        this->data_ = new float[dim];
        fillValues(fill, this->data_, this->total_size);
    }
    
    template<size_t RANK>
    Tensor<RANK>::Tensor( size_t dim, const float fill[] )
    requires (RANK == 1) {
        if( dim == 0 ) {
            throw std::invalid_argument("The dimension size is less than 1.");
        }
        this->dimensions[0] = dim;
        this->total_size = dim;
        this->data_ = new float[dim];
        setValues(fill, this->data_, dim);
    }
    
    template<size_t RANK>
    Tensor<RANK>::Tensor( const size_t (&dims)[RANK] ) {
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
        this->data_ = new float[total_size];
    }
    
    template<size_t RANK>
    Tensor<RANK>::Tensor( const size_t (&dims)[RANK], const float fill ) {
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
        // Allocate memory for data and fill with value
        this->total_size = total_size;
        this->data_ = new float[total_size];
        fillValues(fill, this->data_, this->total_size);
    }
    
    template<size_t RANK>
    Tensor<RANK>::Tensor( InitializerElements<RANK> elements ) {
        if( elements.size() == 0 ) {
            throw std::invalid_argument("The first dimension size is less than 1.");
        }

        // Recursively count the total size of the initializer elements
        this->total_size = countInitializerElements<RANK>(elements, this->dimensions);
        if( !checkInitializerElements<RANK>(elements, this->dimensions) ) {
            throw std::invalid_argument("The given initializer elements are not rectangular");
        }

        // Check if any of the dimensions are 0
        for( size_t i = 0; i < RANK; ++i ) {
            if( this->dimensions[i] < 1 ) {
                throw std::invalid_argument("One or more dimension sizes are less than 1.");
            }
        }

        // Allocate memory
        this->data_ = new float[this->total_size];
        // Assign data from flattened initializer elements
        float* data_ptr = this->data_;
        flattenInitializerElements<RANK>(elements, data_ptr);
    }
    
    template<size_t RANK>
    Tensor<RANK>::Tensor( std::initializer_list<std::reference_wrapper<const BaseTensor<RANK-1>>> elements ) 
    requires (RANK > 1) {
        const size_t dim1 = elements.size();
        if( dim1 == 0 ) {
            throw std::invalid_argument("The first dimension size is less than 1.");
        }

        const std::reference_wrapper<const BaseTensor<RANK-1>>* tensor_refs = elements.begin();
        // Copy dimensions from the first Tensor
        this->dimensions[0] = dim1;
        for( size_t i = 1; i < RANK; ++i ) {
            this->dimensions[i] = tensor_refs[0].get().dimensions[i-1];
        }
        // Check that all Tensors have the same dimensions
        for( size_t i = 1; i < dim1; ++i ) {
            for( size_t j = 0; j < RANK; ++j ) {
                if( this->dimensions[j+1] != tensor_refs[i].get().dimensions[j] ) {
                    throw std::invalid_argument("Two or more dimension sizes do not match.");
                }
            }
        }
        // Allocate memory for data
        const size_t inner_tensor_size = tensor_refs[0].get().total_size;
        this->total_size = dim1 * inner_tensor_size;
        this->data_ = new float[this->total_size];
        // Copy data from Tensors into this
        for( size_t i = 0; i < dim1; ++i ) {
            setValues(tensor_refs[i].get().data_, this->data_ + (i * inner_tensor_size), inner_tensor_size);  
        }
    }

    template<size_t RANK>
    Tensor<RANK>::Tensor( const Tensor<RANK>& other ) {
        // Copy dimensions
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        // Allocate new memory and copy it over
        this->total_size = other.total_size;
        this->data_ = new float[other.total_size];
        setValues(other.data_, this->data_, this->total_size);
    }
    
    template<size_t RANK>
    Tensor<RANK>::Tensor( const BaseTensor<RANK>& other ) {
        // Copy dimensions
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        // Allocate new memory and copy it over
        this->total_size = other.total_size;
        this->data_ = new float[other.total_size];
        setValues(other.data_, this->data_, this->total_size);
    }
    
    template<size_t RANK>
    Tensor<RANK>::Tensor( Tensor<RANK>&& other ) {
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
    
    template<size_t RANK>
    Tensor<RANK>::~Tensor() {
        delete[] this->data_;
    }
    
    template<size_t RANK>
    Tensor<RANK>& Tensor<RANK>::operator = ( const Tensor<RANK>& other ) {
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
            this->data_ = new float[other.total_size];
            this->total_size = other.total_size;
        }
        // Copy over data
        setValues(other.data_, this->data_, this->total_size);

        return *this;
    }
    
    template<size_t RANK>
    Tensor<RANK>& Tensor<RANK>::operator = ( const BaseTensor<RANK>& other ) {
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
            this->data_ = new float[other.total_size];
            this->total_size = other.total_size;
        }
        // Copy over data
        setValues(other.data_, this->data_, this->total_size);

        return *this;
    }
    
    template<size_t RANK>
    Tensor<RANK>& Tensor<RANK>::operator = ( Tensor<RANK>&& other ) {
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
    
    template<size_t RANK>
    Tensor<2> Tensor<RANK>::identity( const size_t dims, const float diagonal_value ) {
        Tensor<2> result({dims, dims}, 0);
        for( size_t i = 0; i < dims; ++i ) {
            result[{i, i}] = diagonal_value;
        }
        return result;
    }


    /* VTensor Implementation */

    template<size_t RANK>
    VTensor<RANK>::VTensor() {
        // Set all dimensions to 0
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = 0;
        }
        // Allocate no memory
        this->total_size = 0;
        this->data_ = nullptr;
    }
    
    template<size_t RANK>
    VTensor<RANK>::VTensor( const VTensor<RANK>& other ) {
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        this->total_size = other.total_size;
        this->data_ = other.data_;
    }
    
    template<size_t RANK>
    VTensor<RANK>& VTensor<RANK>::operator = ( const VTensor<RANK>& other ) {
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        this->total_size = other.total_size;
        this->data_ = other.data_;

        return *this;
    }


    /* RaggedTensor Implementation */

    template<size_t RANK>
    RaggedTensor<RANK>::RaggedTensor() {
        this->total_size = 0;
        this->data_ = nullptr;
        this->dimension1 = 0;
        this->inner_tensors = nullptr;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK>::RaggedTensor( const size_t dim1_size, const size_t inner_tensor_dims[] ) 
    requires (RANK == 2) {
        // Allocate space for array of inner VTensors
        this->inner_tensors = new VTensor<1>[dim1_size];
        // Copy over dimension sizes and count the total size
        size_t total_size = 0;
        for( size_t i = 0; i < dim1_size; ++i ) {
            const size_t inner_tensor_size = inner_tensor_dims[i];
            // Copy over dimension size
            this->inner_tensors[i].dimensions[0] = inner_tensor_size;
            // Set inner tensor total size
            this->inner_tensors[i].total_size = inner_tensor_size;

            total_size += inner_tensor_size;
        }
        this->total_size = total_size;

        // Allocate space for data
        this->data_ = new float[total_size];

        // Set first dimension size
        this->dimension1 = dim1_size;
        // Set the starting position of each inner Tensor
        float* starting_pos = this->data_;
        for( size_t i = 0; i < dim1_size; ++i ) {
            this->inner_tensors[i].data_ = starting_pos;
            starting_pos += this->inner_tensors[i].total_size;
        }
    }
    
    template<size_t RANK>
    RaggedTensor<RANK>::RaggedTensor( const size_t dim1_size, const size_t inner_tensor_dims[][RANK-1] ) {
        // Allocate space for array of inner VTensors
        this->inner_tensors = new VTensor<RANK-1>[dim1_size];
        // Copy over dimension sizes and count the total size
        size_t total_size = 0;
        for( size_t i = 0; i < dim1_size; ++i ) {
            size_t inner_tensor_size = 1;
            // Copy over dimension sizes
            for( size_t j = 0; j < RANK-1; ++j ) {
                const size_t dim = inner_tensor_dims[i][j];
                this->inner_tensors[i].dimensions[j] = dim;
                inner_tensor_size *= dim;
            }
            // Set inner tensor total size
            this->inner_tensors[i].total_size = inner_tensor_size;

            total_size += inner_tensor_size;
        }
        this->total_size = total_size;

        // Allocate space for data
        this->data_ = new float[total_size];

        // Set first dimension size
        this->dimension1 = dim1_size;
        // Set the starting position of each inner Tensor
        float* starting_pos = this->data_;
        for( size_t i = 0; i < dim1_size; ++i ) {
            this->inner_tensors[i].data_ = starting_pos;
            starting_pos += this->inner_tensors[i].total_size;
        }
    }
    
    template<size_t RANK>
    RaggedTensor<RANK>::RaggedTensor( std::initializer_list<size_t> inner_tensor_dims )
    requires (RANK == 2)  
        : RaggedTensor<RANK>(inner_tensor_dims.size(), inner_tensor_dims.begin()) { }
    
        template<size_t RANK>
    RaggedTensor<RANK>::RaggedTensor( std::initializer_list<size_t[RANK-1]> inner_tensor_dims )
        : RaggedTensor<RANK>(inner_tensor_dims.size(), inner_tensor_dims.begin()) { }
    
        template<size_t RANK>
    RaggedTensor<RANK>::RaggedTensor( std::initializer_list<std::reference_wrapper<const BaseTensor<RANK-1>>> elements ) {
        const size_t dim1 = elements.size();

        // Allocate space for array of inner VTensors
        this->inner_tensors = new VTensor<RANK-1>[dim1];
        // Copy over dimension sizes and count the total size
        const std::reference_wrapper<const BaseTensor<RANK-1>>* inner_tensor_ptrs = elements.begin();
        size_t total_size = 0;
        for( size_t i = 0; i < dim1; ++i ) {
            // Copy over dimension sizes
            for( size_t j = 0; j < RANK-1; ++j ) {
                const size_t dim = inner_tensor_ptrs[i].get().dimensions[j];
                this->inner_tensors[i].dimensions[j] = dim;
            }
            // Set inner tensor total size
            const size_t inner_tensor_size = this->inner_tensors[i].total_size;
            this->inner_tensors[i].total_size = inner_tensor_size;

            total_size += inner_tensor_size;
        }
        this->total_size = total_size;

        // Allocate space for data
        this->data_ = new float[total_size];

        // Set first dimension size
        this->dimension1 = dim1;
        // Set the starting position of each inner Tensor
        float* starting_pos = this->data_;
        for( size_t i = 0; i < dim1; ++i ) {
            this->inner_tensors[i].data_ = starting_pos;
            this->inner_tensors[i].set(inner_tensor_ptrs[i]);

            starting_pos += this->inner_tensors[i].total_size;
        }
    }

    template<size_t RANK>
    RaggedTensor<RANK>::RaggedTensor( const RaggedTensor<RANK>& other ) {
        this->total_size = other.total_size;
        // Allocate space for data and copy over
        this->data_ = new float[other.total_size];
        setValues(other.data_, this->data_, this->total_size);
        
        const size_t dim1 = other.dimension1;
        this->dimension1 = dim1;
        // Allocate space for inner tensors and copy over
        this->inner_tensors = new VTensor<RANK-1>[dim1];
        float* starting_pos = this->data_;
        for( size_t i = 0; i < dim1; ++i ) {
            this->inner_tensors[i] = other.inner_tensors[i];
            // Set a pointer from the newly allocated data
            this->inner_tensors[i].data_ = starting_pos;
            starting_pos += other.inner_tensors[i].total_size;
        }
    }
    
    template<size_t RANK>
    RaggedTensor<RANK>::RaggedTensor( const BaseTensor<RANK>& other ) {
        this->total_size = other.total_size;
        // Allocate space for data and copy over
        this->data_ = new float[other.total_size];
        setValues(other.data_, this->data_, this->total_size);

        const size_t dim1 = other.dimensions[0];
        this->dimension1 = dim1;
        // Allocate space for inner tensors and copy over
        this->inner_tensors = new VTensor<RANK-1>[dim1];
        const size_t inner_tensor_total_size = other.total_size / dim1;
        float* starting_pos = this->data_;
        for( size_t i = 0; i < dim1; ++i ) {
            // Copy over sizes
            this->inner_tensors[i].total_size = inner_tensor_total_size;
            for( size_t j = 1; j < RANK; ++j ) {
                this->inner_tensors[i].dimensions[j-1] = other.dimensions[j];
            }

            // Set a pointer from the newly allocated data
            this->inner_tensors[i].data_ = starting_pos;
            starting_pos += inner_tensor_total_size;
        }
    }
    
    template<size_t RANK>
    RaggedTensor<RANK>::RaggedTensor( RaggedTensor<RANK>&& other ) {
        // Move data to this RaggedTensor
        this->total_size = other.total_size;
        this->data_ = other->data_;
        this->dimension1 = other->dimension1;
        this->inner_tensors = other.inner_tensors;

        // Clear data from other RaggedTensor
        other.total_size = 0;
        other.data_ = nullptr;
        other.dimension1 = 0;
        other.inner_tensors = nullptr;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK>::~RaggedTensor() {
        delete[] this->inner_tensors;
        delete[] this->data_;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK>& RaggedTensor<RANK>::operator = ( const RaggedTensor<RANK>& other ) {
        // Check for self assignment
        if( this == &other ) {
            return *this;
        }
        // Free the previous data held in this RaggedTensor.
        delete[] this->data_;

        this->total_size = other.total_size;
        // Allocate space for data and copy over
        this->data_ = new float[other.total_size];
        setValues(other.data_, this->data_, this->total_size);

        const size_t dim1 = other.dimension1;
        this->dimension1 = dim1;
        // Allocate space for inner tensors and copy over
        this->inner_tensors = new VTensor<RANK-1>[dim1];
        float* starting_pos = this->data_;
        for( size_t i = 0; i < dim1; ++i ) {
            this->inner_tensors[i] = other.inner_tensors[i];
            // Set a pointer from the newly allocated data
            this->inner_tensors[i].data_ = starting_pos;
            starting_pos += other.inner_tensors[i].total_size;
        }

        return *this;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK>& RaggedTensor<RANK>::operator = ( const BaseTensor<RANK>& other ) {
        // Check for self assignment
        if( this == &other ) {
            return *this;
        }
        // Free the previous data held in this RaggedTensor.
        delete[] this->data_;

        this->total_size = other.total_size;
        // Allocate space for data and copy over
        this->data_ = new float[other.total_size];
        setValues(other.data_, this->data_, this->total_size);

        const size_t dim1 = other.dimensions[0];
        this->dimension1 = dim1;
        // Allocate space for inner tensors and copy over
        this->inner_tensors = new VTensor<RANK-1>[dim1];
        const size_t inner_tensor_total_size = other.total_size / dim1;
        float* starting_pos = this->data_;
        for( size_t i = 0; i < dim1; ++i ) {
            // Copy over sizes
            this->inner_tensors[i].total_size = inner_tensor_total_size;
            for( size_t j = 1; j < RANK; ++j ) {
                this->inner_tensors[i].dimensions[j-1] = other.dimensions[j];
            }

            // Set a pointer from the newly allocated data
            this->inner_tensors[i].data_ = starting_pos;
            starting_pos += inner_tensor_total_size;
        }

        return *this;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK>& RaggedTensor<RANK>::operator = ( RaggedTensor<RANK>&& other ) {
        // Check for self assignment
        if( this == &other ) {
            return *this;
        }
        // Free the previous data held in this RaggedTensor.
        delete[] this->data_;

        // Move data to this RaggedTensor
        this->total_size = other.total_size;
        this->data_ = other->data_;
        this->dimension1 = other->dimension1;
        this->inner_tensors = other.inner_tensors;

        // Clear data from other RaggedTensor
        other.total_size = 0;
        other.data_ = nullptr;
        other.dimension1 = 0;
        other.inner_tensors = nullptr;

        return *this;
    }

    template<size_t RANK>
    const float& RaggedTensor<RANK>::operator [] ( const size_t (&indexes)[RANK] ) const {
        return this->inner_tensors[indexes[0]][indexes + 1];
    }
    
    template<size_t RANK>
    float& RaggedTensor<RANK>::operator [] ( const size_t (&indexes)[RANK] ) {
        return this->inner_tensors[indexes[0]][indexes + 1];
    }
    
    template<size_t RANK>
    const VTensor<RANK-1> RaggedTensor<RANK>::operator [] ( size_t index ) const {
        return this->inner_tensors[index];
    }
    
    template<size_t RANK>
    VTensor<RANK-1> RaggedTensor<RANK>::operator [] ( size_t index ) {
        return this->inner_tensors[index];
    }

    template<size_t RANK>
    RaggedTensor<RANK> RaggedTensor<RANK>::operator + ( const RaggedTensor<RANK>& other ) const {
        RaggedTensor<RANK> result = this->emptied();
        // Add the values in the Tensors together
        addValues(this->data_, other.data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK> RaggedTensor<RANK>::operator - ( const RaggedTensor<RANK>& other ) const {
        RaggedTensor<RANK> result = this->emptied();
        // Subtract the values in this Tensor by the values in the other Tensor
        subtractValues(this->data_, other.data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK> RaggedTensor<RANK>::operator * ( const RaggedTensor<RANK>& other ) const {
        RaggedTensor<RANK> result = this->emptied();
        // Multiply the values in the Tensors together
        multiplyValues(this->data_, other.data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK> RaggedTensor<RANK>::operator / ( const RaggedTensor<RANK>& other ) const {
        RaggedTensor<RANK> result = this->emptied();
        // Divide the values in this Tensor by the values in the other Tensor
        divideValues(this->data_, other.data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK> operator * ( const RaggedTensor<RANK>& tensor, const float scale ) {
        RaggedTensor<RANK> result = tensor.emptied();
        // Multiply the values in the Tensor by the scale
        multiplyValuesByScalar(tensor.data_, scale, result.data_, tensor.total_size);

        return result;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK> operator * ( const float scale, const RaggedTensor<RANK>& tensor ) {
        RaggedTensor<RANK> result = tensor.emptied();
        // Multiply the values in the Tensor by the scale
        multiplyValuesByScalar(tensor.data_, scale, result.data_, tensor.total_size);
        
        return result;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK> RaggedTensor<RANK>::operator / ( const float scale ) const {
        RaggedTensor<RANK> result = this->emptied();
        // Multiply the values in this Tensor by the inverted scale
        multiplyValuesByScalar(this->data_, (1.0f / scale), result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK> RaggedTensor<RANK>::operator - () const {
        RaggedTensor<RANK> result = this->emptied();
        // Negate the values in this Tensor
        negateValues(this->data_, result.data_, this->total_size);

        return result;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK> RaggedTensor<RANK>::emptied() const {
        size_t inner_tensor_dims[this->dimension1][RANK-1];
        // Copy over dimensions
        for( size_t i = 0; i < this->dimension1; ++i ) {
            for( size_t j  = 0; j < RANK-1; ++j ) {
                inner_tensor_dims[i][j] = this->inner_tensors[i].dimensions[j];
            }
        }
        return RaggedTensor<RANK>(this->dimension1, inner_tensor_dims);
    } 
    
    template<size_t RANK>
    bool RaggedTensor<RANK>::isSameSize( const RaggedTensor<RANK>& other ) const {
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
    
    template<size_t RANK>
    bool RaggedTensor<RANK>::operator == ( const RaggedTensor<RANK>& other ) const {
        if( !this->isSameSize(other) ) {
            return false;
        }

        return compareValuesForEquality(this->data_, other.data_, this->total_size);
    }
    
    template<size_t RANK>
    bool RaggedTensor<RANK>::operator != ( const RaggedTensor<RANK>& other ) const {
        return !(*this == other);
    }

    template<size_t RANK>
    RaggedTensor<RANK>& RaggedTensor<RANK>::operator += ( const RaggedTensor<RANK>& other ) {
        // Add other's values
        addValues(this->data_, other.data_, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK>& RaggedTensor<RANK>::operator -= ( const RaggedTensor<RANK>& other ) {
        // Subtract other's values
        subtractValues(this->data_, other.data_, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK>& RaggedTensor<RANK>::operator *= ( const RaggedTensor<RANK>& other ) {
        // Add other's values
        multiplyValues(this->data_, other.data_, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK>& RaggedTensor<RANK>::operator /= ( const RaggedTensor<RANK>& other ) {
        // Subtract other's values
        divideValues(this->data_, other.data_, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK>& RaggedTensor<RANK>::operator *= ( const float scale ) {
        // Multiply by scale
        multiplyValuesByScalar(this->data_, scale, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK>& RaggedTensor<RANK>::operator /= ( const float scale ) {
        // Multiply by scale
        multiplyValuesByScalar(this->data_, 1.0f / scale, this->data_, this->total_size);
        return *this;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK>& RaggedTensor<RANK>::fill( const float fill ) {
        fillValues(fill, this->data, this->total_size);
        return *this;
    }
    
    template<size_t RANK>
    RaggedTensor<RANK>& RaggedTensor<RANK>::set( const RaggedTensor<RANK>& tensor ) {
        setValues(tensor.data_, this->data_, this->total_size);
        return *this;
    }
    
    /* TODO: Add support for indexing */
    template<size_t RANK>
    template<typename Func>
    RaggedTensor<RANK>& RaggedTensor<RANK>::transform( Func transform_function ) {
        // If the function takes no arguments
        if constexpr( std::is_invocable_r_v<float, Func> ) {
            for( size_t i = 0; i < this->total_size; ++i ) {
                this->data_[i] = transform_function();
            }
            return *this;
        }

        // If the function just takes the values, but not any indexes
        else if constexpr( std::is_invocable_r_v<float, Func, float> ) {
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

    template<size_t RANK>
    size_t RaggedTensor<RANK>::rank() const {
        return RANK;
    }
    
    template<size_t RANK>
    size_t RaggedTensor<RANK>::totalSize() const {
        return this->total_size;
    }
    
    template<size_t RANK>
    size_t RaggedTensor<RANK>::dim1Size() const {
        return this->dimension1;
    }
    
    template<size_t RANK>
    const float* RaggedTensor<RANK>::data() const {
        return this->data_;
    }
}



#endif