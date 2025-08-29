/* Tensor.hpp */

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <initializer_list>
#include <cassert>



// Declaration
namespace jai {
    /* 
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

    /* 
     * Recursive type used to initialize a Tensor of rank `RANK` with elements.
     * An InitializerElements<RANK> contains a set of InitializerElements<RANK-1>s,
     * and an InitializerElements<1> contains a set of floats.
     */
    template <size_t RANK>
    using InitializerElements = typename InitializerElementsType<RANK>::type;

    /* Tensor Declarations */
    template<size_t RANK>
    class VTensor;
    template<size_t RANK>
    class Tensor;


    /**
     * This represents the view into a part or whole of a Tensor.
     * Any instance of a VTensor, and the data contained within, is backed by a Tensor.
     * Despite its name, a VTensor can be modified, but it will also modify the Tensor it is backed by.
     */
    template<size_t RANK>
    class VTensor {
        // Ensure that Tensor RANK cannot be 0 (must have 1 or more dimensions)
        static_assert(RANK > 0, "Tensor rank cannot be 0.");

        /* Constructors */
        public:

        VTensor();
        VTensor( const VTensor<RANK>& other );
        VTensor<RANK>& operator = ( const VTensor<RANK>& other );
        VTensor( VTensor<RANK>&& other );
        VTensor<RANK>& operator = ( VTensor<RANK>&& other );

        /*
         * Disallow slicing from a Tensor to a VTensor.
         * We are not allowing Tensors to be casted to VTensors, because it can cause an
         * issue with reassigning internal data, and therefore cause a memory leak.
         * Tensor::view() can be used to safely retrieve a view of an entire Tensor.
         * Important to remember that VTensors only provide a view of a Tensor;
         * the assigned VTensor is still dependent on the assigning Tensor.
         */
        VTensor( const Tensor<RANK>& ) = delete;
        VTensor<RANK>& operator = ( Tensor<RANK>&& ) = delete;

        /* Accessors */
        public:

        /** 
         * Defined for RANK=1 Tensors, this returns the element at the given index in the first (and only) dimension.
         */
        template<size_t R = RANK, typename std::enable_if<(R == 1), int>::type = 0>
        const float& operator [] ( size_t index ) const;
        /**
         * Defined for RANK=1 Tensors, this returns a mutable reference to the element at the `index` in the first (and only) dimension.
         */
        template<size_t R = RANK, typename std::enable_if<(R == 1), int>::type = 0>
        float& operator [] ( size_t index );
        /* Defined for RANK>1 Tensors, returns the element at the given indexes.
         */
        template<size_t R = RANK, typename std::enable_if<(R > 1), int>::type = 0>
        const float& operator [] ( const size_t (&indexes)[RANK] ) const;
        /* Defined for RANK>1 Tensors, this returns a mutable reference to the element at the given `indexes`.
         */
        template<size_t R = RANK, typename std::enable_if<(R > 1), int>::type = 0>
        float& operator [] ( const size_t (&indexes)[RANK] );
        /* Defined for RANK>1 Tensors, this returns an immutable View Tensor with rank RANK-1, at the given index in the first dimension.
         * The returned View Tensor is backed by `this` Tensor.
         */
        template<size_t R = RANK, typename std::enable_if<(R > 1), int>::type = 0>
        const VTensor<RANK-1> operator [] ( size_t index ) const;
        /* Defined for RANK>1 Tensors, this returns a View Tensor with rank RANK-1, at the given index in the first dimension.
         * The returned View Tensor is backed by `this` Tensor.
         */
        template<size_t R = RANK, typename std::enable_if<(R > 1), int>::type = 0>
        VTensor<RANK-1> operator [] ( size_t index );

        /* Returns an immutable View Tensor which is backed by `this` Tensor.
         */
        const VTensor<RANK> view() const;
        /* Returns a View Tensor which is backed by `this` Tensor.
         */
        VTensor<RANK> view();
        /* Returns an immutable View Tensor with rank RANK+1 of `this` Tensor, where it's last dimension is of size 1.
         * The returned View Tensor is backed by `this` Tensor.
         * Useful for converting a Vector into an (n x 1) Matrix for matrix multiplication.
         */
        const VTensor<RANK+1> rankUp() const;
        /* Returns a View Tensor with rank RANK+1 of `this` Tensor, where it's last dimension is of size 1.
         * The returned View Tensor is backed by `this` Tensor.
         * Useful for converting a Vector into an (n x 1) Matrix for matrix multiplication.
         */
        VTensor<RANK+1> rankUp();
        /* Returns an immutable View Tensor with rank 1 of `this` Tensor, all of it's values are flattened into one vector.
         * The returned View Tensor is backed by `this` Tensor.
         */
        const VTensor<1> flattened() const;
        /* Returns a View Tensor with rank 1 of `this` Tensor, all of it's values are flattened into one vector.
         * The returned View Tensor is backed by `this` Tensor.
         */
        VTensor<1> flattened();

        /* Binary Operations */
        public:

        /* Adds all of the elements in the `other` Tensor to all of the elements in `this` Tensor and returns the result.
         * Both Tensors must be the same total size, but do not necessarily have to have the same dimensions.
         * The dimensions of this Tensor are passed onto the result Tensor.
         */
        Tensor<RANK> operator + ( const VTensor<RANK>& other ) const;
        /* Subtracts all of the elements in the `other` Tensor from all of the elements in `this` Tensor and returns the result.
         * Both Tensors must be the same total size, but do not necessarily have to have the same dimensions.
         * The dimensions of this Tensor are passed onto the result Tensor.
         */
        Tensor<RANK> operator - ( const VTensor<RANK>& other ) const;
        /* Multiplies all of the elements in `this` Tensor by `scale` and returns the result.
        */
        template<size_t R>
        friend Tensor<R> operator * ( const VTensor<R>& tensor, float scale );
        /* Multiplies all of the elements in `this` Tensor by `scale` and returns the result.
        */
        template<size_t R>
        friend Tensor<R> operator * ( float scale, const VTensor<R>& tensor );
        /* Divides all of the elements in `this` Tensor by `scale` and returns the result.
        */
        Tensor<RANK> operator / ( float scale ) const;
        /* Negates all of the elements in `this` Tensor and returns the result.
         */
        Tensor<RANK> operator - () const;

        /* General mutators */
        public:

        /* This sets the values in `this` Tensor to the values in `tensor`.
         * The given `tensor` must have the same dimensions as `this` Tensor.
         */
        void set( const VTensor<RANK>& tensor );
        /* Adds all of the elements in the other Tensor to all of the elements in this Tensor.
         * The other Tensor must be the same total size as this Tensor, but does not necessarily have to have the same dimensions.
         */
        void addTo( const VTensor<RANK>& other );
        /* Subtracts all of the elements in the other Tensor from all of the elements in this Tensor.
         * The other Tensor must be the same total size as this Tensor, but does not necessarily have to have the same dimensions.
         */
        void subFrom( const VTensor<RANK>& other );
        /* Multiples all of the elements in this Tensor with the given scale.
         */
        void scaleBy( float scale );

        /* Vector operations */
        public:

        /* Finds the magnitude of this Vector and returns the result.
         */
        template<size_t R = RANK, typename std::enable_if<(R == 1), int>::type = 0>
        float mag() const;
        /* Finds the squared magnitude of this Vector and returns the result.
         */
        template<size_t R = RANK, typename std::enable_if<(R == 1), int>::type = 0>
        float squaredMag() const;
        /* Takes the dot product of this Vector with the other Vector and returns the result.
         * The two vectors must be the same size.
         */
        template<size_t R = RANK, typename std::enable_if<(R == 1), int>::type = 0>
        float dot( const VTensor<1>& other ) const;
        /* Takes the cross product of this Vector with the other Vector and returns the result.
         * The two vectors must have a size of 3.
         */
        template<size_t R = RANK, typename std::enable_if<(R == 1), int>::type = 0>
        Tensor<1> cross( const VTensor<1>& other ) const;

        /* Matrix operations */
        public:

        /* Finds the matrix multiplication of the other Matrix on this Matrix and returns the result.
         * This Matrix must be of size (m x n) and the other Matrix must be of size (n x w)
         */
        template<size_t R = RANK, typename std::enable_if<(R == 2), int>::type = 0>
        Tensor<2> mul( const VTensor<2>& other ) const;
        /* Finds the matrix multiplication of the other Vector on this Matrix and returns the result.
         * This matrix must be of size (m x n) and the other Vector must be of size (n).
         */
        template<size_t R = RANK, typename std::enable_if<(R == 2), int>::type = 0>
        Tensor<1> mul( const VTensor<1>& other ) const;
        
        /* Transposes this Matrix.
         */
        template<size_t R = RANK, typename std::enable_if<(R == 2), int>::type = 0>
        void transpose();

        /* Getters */
        public:

        /* Returns the rank of the tensor (the number of dimensions).
         * NOTE: This is NOT the matrix rank.
         */
        size_t rank() const;
        /* Defined for RANK=1 Tensors, this returns the size of the Tensor.
         * This is the same as calling totalSize()
         */
        template<size_t R = RANK, typename std::enable_if<(R == 1), int>::type = 0>
        size_t size() const;
        /* Defined for RANK>1 Tensors, this returns the size of the given dimension.
         */
        template<size_t R = RANK, typename std::enable_if<(R > 1), int>::type = 0>
        size_t size( size_t dimension ) const;
        /* Returns the total size of the Tensor (the total number of elements).
         */
        size_t totalSize() const;

        /* Returns true if the `other` Tensor has the same dimensions as `this` Tensor, and false otherwise.
         */
        bool isSameSize( const VTensor<RANK>& other ) const;
        /**
         * Returns true if the `other` Tensor is equal to `this` Tensor, and false otherwise.
         */
        bool operator == ( const VTensor<RANK>& other ) const;
        /**
         * Returns true if the `other` Tensor is not equal to `this` Tensor, and false otherwise.
         */
        bool operator != ( const VTensor<RANK>& other ) const;

        /* Prints out the Tensor as a string.
         */
        template<size_t R>
        friend std::ostream& operator << ( std::ostream& fs, const VTensor<R>& t );

        protected:
        public:
        size_t dimensions[RANK];
        size_t total_size;
        float* data;
    };

    
    /** 
     * This represents a Tensor itself, which contains and manages all of it's own data.
     * Any instance of a Tensor, and the data contained within, is managed by itself.
     */
    template<size_t RANK>
    class Tensor :  public VTensor<RANK> {
        /* Constructors */
        public:

        /* Constructs an empty Tensor with a size of 0 in each dimension.
         */
        Tensor();
        /* Defined for RANK=1 Tensors, constructs a Tensor with the given dimension.
         * Throws an error if `dim` is equal to 0.
         */
        template<size_t R = RANK, typename std::enable_if<(R == 1), int>::type = 0>
        Tensor( size_t dim );
        /* Defined for RANK=1 Tensors, constructs a Tensor with the given dimensions and with all values set to `fill`.
         * Throws an error if `dim` is equal to 0.
         */
        template<size_t R = RANK, typename std::enable_if<(R == 1), int>::type = 0>
        Tensor( size_t dim, float fill );
        /* Defined for RANK>1 Tensors, constructs a Tensor with the given dimensions.
         * Throws an error if any value in `dims` is equal to 0.
         */
        template<size_t R = RANK, typename std::enable_if<(R > 1), int>::type = 0>
        Tensor( const size_t (&dims)[RANK] );
        /* Defined for RANK>1 Tensors, constructs a Tensor with the given dimensions and with all values set to `fill`.
         * Throws an error if any value in `dims` is equal to 0.
         */
        template<size_t R = RANK, typename std::enable_if<(R > 1), int>::type = 0>
        Tensor( const size_t (&dims)[RANK], float fill );
        /* Constructs a Tensor initialized with the given `elements`.
         * Throws an error if `elements` or any inner elements inside `elements` has a size of 0.
         * Throws an error if the `elements` are non-rectangular.
         */
        Tensor( InitializerElements<RANK> elements );
        /* Defined for RANK>1 Tensors, constructs a Tensor initialized with the given `Tensor<RANK-1>` elements.
         * The size of the first dimension is the size of `elements`.
         * Throws an error if `elements` has a size of 0.
         * Throws an error if any of the Tensors in `elements` have differing dimensions.
         */
        template<size_t R = RANK, typename std::enable_if<(R > 1), int>::type = 0>
        Tensor( std::initializer_list<std::reference_wrapper<const VTensor<RANK-1>>> elements );

        /* Copy constructor.
         */
        Tensor( const Tensor<RANK>& other );
        /* Copy constructor from VTensor.
         */
        Tensor( const VTensor<RANK>& other );
        /* Move constructor.
         */
        Tensor( Tensor<RANK>&& other );
        /* Destructor.
         */
        ~Tensor();
        /* Assignment operator. Ensures that memory is freed when existing object is overwritten.
         */
        Tensor<RANK>& operator = ( const Tensor<RANK>& other );
        /* Assignment operator from VTensor. Ensures that memory is freed when existing object is overwritten.
         */
        Tensor<RANK>& operator = ( const VTensor<RANK>& other );
        /* Move assignment operator. Ensures that memory is freed when existing object is overwritten.
         */
        Tensor<RANK>& operator = ( Tensor<RANK>&& other );

        /* Helper functions */
        private:
        public:
        static size_t countInitializerElements( const InitializerElements<RANK>& elements, size_t dims[RANK] );
        static bool checkInitializerElements( const InitializerElements<RANK>& elements, const size_t dims[RANK] );
        static void flattenInitializerElements( const InitializerElements<RANK>& elements, float*& data );
    };


    /* Vector and Matrix type definitions */
    //using Vector = VTensor<1>;
    //using Matrix = VTensor<2>;
}



// Implementation
namespace jai {
    /* VTensor Implementation */

    template<size_t RANK>
    VTensor<RANK>::VTensor() {
        // Set all dimensions to 0
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = 0;
        }
        // Allocate no memory
        this->total_size = 0;
        this->data = nullptr;
    }
    template<size_t RANK>
    VTensor<RANK>::VTensor( const VTensor<RANK>& other ) {
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        this->total_size = other.total_size;
        this->data = other.data;
    }
    template<size_t RANK>
    VTensor<RANK>& VTensor<RANK>::operator = ( const VTensor<RANK>& other ) {
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        this->total_size = other.total_size;
        this->data = other.data;

        return *this;
    }
    template<size_t RANK>
    VTensor<RANK>::VTensor( VTensor<RANK>&& other ) {
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        this->total_size = other.total_size;
        this->data = other.data;
    }
    template<size_t RANK>
    VTensor<RANK>& VTensor<RANK>::operator = ( VTensor<RANK>&& other ) {
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        this->total_size = other.total_size;
        this->data = other.data;

        return *this;
    }

    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 1), int>::type>
    const float& VTensor<RANK>::operator [] ( const size_t index ) const {
        return this->data[index];
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 1), int>::type>
    float& VTensor<RANK>::operator [] ( const size_t index ) {
        return this->data[index];
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R > 1), int>::type>
    const float& VTensor<RANK>::operator [] ( const size_t (&indexes)[RANK] ) const {
        size_t index = 0;
        size_t inner_tensor_total_size = this->total_size;
        for( size_t i = 0; i < RANK; ++i ) {
            inner_tensor_total_size /= this->dimensions[i];
            index += inner_tensor_total_size * indexes[i];
        }
        return this->data[index];
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R > 1), int>::type>
    float& VTensor<RANK>::operator [] ( const size_t (&indexes)[RANK] ) {
        size_t index = 0;
        size_t inner_tensor_size = this->total_size;
        for( size_t i = 0; i < RANK; ++i ) {
            inner_tensor_size /= this->dimensions[i];
            index += inner_tensor_size * indexes[i];
        }
        return this->data[index];
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R > 1), int>::type>
    const VTensor<RANK-1> VTensor<RANK>::operator [] ( const size_t index ) const {
        VTensor<RANK-1> inner_view;
        for( size_t i = 0; i < RANK-1; ++i ) {
            inner_view.dimensions[i] = this->dimensions[i+1];
        }
        const size_t inner_tensor_total_size = this->total_size / this->dimensions[0];
        inner_view.total_size = inner_tensor_total_size;
        inner_view.data = this->data + inner_tensor_total_size*index;
        return inner_view;
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R > 1), int>::type>
    VTensor<RANK-1> VTensor<RANK>::operator [] ( const size_t index ) {
        VTensor<RANK-1> inner_view;
        for( size_t i = 0; i < RANK-1; ++i ) {
            inner_view.dimensions[i] = this->dimensions[i+1];
        }
        const size_t inner_tensor_total_size = this->total_size / this->dimensions[0];
        inner_view.total_size = inner_tensor_total_size;
        inner_view.data = this->data + inner_tensor_total_size*index;
        return inner_view;
    }

    template<size_t RANK>
    const VTensor<RANK> VTensor<RANK>::view() const {
        return VTensor<RANK>(*this);
    }
    template<size_t RANK>
    VTensor<RANK> VTensor<RANK>::view() {
        return VTensor<RANK>(*this);
    }
    template<size_t RANK>
    const VTensor<RANK+1> VTensor<RANK>::rankUp() const {
        VTensor<RANK+1> ranked_up_view;
        for( size_t i = 0; i < RANK; ++i ) {
            ranked_up_view.dimensions[i] = this->dimensions[i];
        }
        ranked_up_view.dimensions[RANK] = 1;
        ranked_up_view.total_size = this->total_size;
        ranked_up_view.data = this->data;
        return ranked_up_view;
    }
    template<size_t RANK>
    VTensor<RANK+1> VTensor<RANK>::rankUp() {
        VTensor<RANK+1> ranked_up_view;
        for( size_t i = 0; i < RANK; ++i ) {
            ranked_up_view.dimensions[i] = this->dimensions[i];
        }
        ranked_up_view.dimensions[RANK] = 1;
        ranked_up_view.total_size = this->total_size;
        ranked_up_view.data = this->data;
    }
    template<size_t RANK>
    const VTensor<1> VTensor<RANK>::flattened() const {
        VTensor<1> flattened_view;
        flattened_view.dimensions[0] = this->total_size;
        flattened_view.total_size = this->total_size;
        flattened_view.data = this->data;
        return flattened_view;
    }
    template<size_t RANK>
    VTensor<1> VTensor<RANK>::flattened() {
        VTensor<1> flattened_view;
        flattened_view.dimensions[0] = this->total_size;
        flattened_view.total_size = this->total_size;
        flattened_view.data = this->data;
        return flattened_view;
    }

    template<size_t RANK>
    Tensor<RANK> VTensor<RANK>::operator + ( const VTensor<RANK>& other ) const {
        // Copy this to new Tensor and add other to it
        Tensor<RANK> result(*this);
        result.addTo(other);
        // Return result Tensor
        return result;
    }
    template<size_t RANK>
    Tensor<RANK> VTensor<RANK>::operator - ( const VTensor<RANK>& other ) const {
        // Copy this to new Tensor and subtract other from it
        Tensor<RANK> result(*this);
        result.subFrom(other);
        // Return result Tensor
        return result;
    }
    template<size_t RANK>
    Tensor<RANK> operator * ( const VTensor<RANK>& tensor, const float scale ) {
        // Copy this to new Tensor and multiply it by scale
        Tensor<RANK> result(tensor);
        result.scaleBy(scale);
        // Return result Tensor
        return result;
    }
    template<size_t RANK>
    Tensor<RANK> operator * ( const float scale, const VTensor<RANK>& tensor ) {
        // Copy this to new Tensor and multiply it by scale
        Tensor<RANK> result(tensor);
        result.scaleBy(scale);
        // Return result Tensor
        return result;
    }
    template<size_t RANK>
    Tensor<RANK> VTensor<RANK>::operator / ( const float scale ) const {
        // Copy this to new Tensor and divide it by scale
        Tensor<RANK> result(*this);
        result.scaleBy(1.0f / scale);
        // Return result Tensor
        return result;
    }
    template<size_t RANK>
    Tensor<RANK> VTensor<RANK>::operator - () const {
        // Copy this to new Tensor
        Tensor<RANK> result(*this);
        // Negate all elements in result and return it
        for( int i = 0; i < this->total_size; ++i ) {
            result.data[i] *= -1;
        }
        // Return result Tensor
        return result;
    }

    template<size_t RANK>
    void VTensor<RANK>::set( const VTensor<RANK>& tensor ) {
        std::memcpy( this->data, tensor.data, this->total_size * sizeof(float) );
    }
    template<size_t RANK>
    void VTensor<RANK>::addTo( const VTensor<RANK>& other ) {
        // Add other's values
        for( size_t i = 0; i < this->total_size; ++i ) {
            this->data[i] += other.data[i];
        }
    }
    template<size_t RANK>
    void VTensor<RANK>::subFrom( const VTensor<RANK>& other ) {
        // Subtract other's values
        for( size_t i = 0; i < this->total_size; ++i ) {
            this->data[i] -= other.data[i];
        }
    }
    template<size_t RANK>
    void VTensor<RANK>::scaleBy( const float scale ) {
        // Multiply by scale
        for( size_t i = 0; i < this->total_size; ++i ) {
            this->data[i] *= scale;
        }
    }

    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 1), int>::type>
    float VTensor<RANK>::mag() const {
        return std::sqrt(this->squaredMag());
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 1), int>::type>
    float VTensor<RANK>::squaredMag() const {
        float sqrd_sum = 0;
        for( size_t i = 0; i < this->dimensions[0]; ++i ) {
            sqrd_sum += data[i] * data[i];
        }
        return sqrd_sum;
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 1), int>::type>
    float VTensor<RANK>::dot( const VTensor<1>& other ) const {
        float sum = 0;
        for( size_t i = 0; i < this->dimensions[0]; ++i ) {
            sum += this->data[i] * other.data[i];
        }
        return sum;
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 1), int>::type>
    Tensor<1> VTensor<RANK>::cross( const VTensor<1>& other ) const {
        Tensor<1> result(3);
        result[0] = this[1] * other[2] - this[2] * other[1];
        result[1] = this[2] * other[0] - this[0] * other[2];
        result[2] = this[0] * other[1] - this[1] * other[0];
        return result;
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 2), int>::type>
    Tensor<2> VTensor<RANK>::mul( const VTensor<2>& other ) const {
        // Create result Tensor
        Tensor<2> result({this->dimensions[0], other.dimensions[1]});
        // Perform matrix multiplication
        for( size_t i = 0; i < result.dimensions[0]; ++i ) {
            for( size_t j = 0; j < result.dimensions[1]; ++j ) {
                float sum = 0;
                for( size_t k = 0; k < this->dimensions[1]; ++k ) {
                    sum += this[{i, k}] * other[{k, j}];
                }
                result[{i, j}] = sum;
            }
        }
        return result;
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 2), int>::type>
    Tensor<1> VTensor<RANK>::mul( const VTensor<1>& other ) const {
        // Create result Tensor
        Tensor<1> result(this->dimensions[0]);
        // Perform matrix multiplication
        for( size_t i = 0; i < result.dimensions[0]; ++i ) {
            float sum = 0;
            for( size_t j = 0; j < result.dimensions[1]; ++j ) {
                sum += this[{j, i}] * other[i];
            }
            result[i] = sum;
        }
        return result;
    }

    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 2), int>::type>
    void VTensor<RANK>::transpose() {
        for( int i = 0; i < this->dimensions[0]; ++i ) {
            for( int j = i + 1; j < this->dimensions[1]; ++j ) {
                const float temp = (*this)[{i, j}];
                (*this)[{i, j}] = (*this)[{j, i}];
                (*this)[{j, i}] = temp;
            }
        }
    }

    template<size_t RANK>
    size_t VTensor<RANK>::rank() const {
        return RANK;
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 1), int>::type>
    size_t VTensor<RANK>::size() const {
        return this->total_size;
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R > 1), int>::type>
    size_t VTensor<RANK>::size( const size_t dimension ) const {
        return this->dimensions[dimension];
    }
    template<size_t RANK>
    size_t VTensor<RANK>::totalSize() const {
        return this->total_size;
    }

    template<size_t RANK>
    bool VTensor<RANK>::isSameSize( const VTensor<RANK>& other ) const {
        for( size_t i = 0; i < RANK; ++i ) {
            if( this->dimensions[i] != other.dimensions[i] ) {
                return false;
            }
        }
        return true;
    }
    template<size_t RANK>
    bool VTensor<RANK>::operator == ( const VTensor<RANK>& other ) const {
        if( this->isSameSize(other) ) {
            return false;
        }

        for( size_t i = 0; i < this->total_size; ++i ) {
            if( this->data[i] != other.data[i] ) {
                return false;
            }
        }

        return true;
    }
    template<size_t RANK>
    bool VTensor<RANK>::operator != ( const VTensor<RANK>& other ) const {
        return !(*this == other);
    }

    template<size_t RANK>
    std::ostream& operator << ( std::ostream& fs, const VTensor<RANK>& t ) {
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


    /* Tensor Implementation */

    template<size_t RANK>
    Tensor<RANK>::Tensor() {
        // Set all dimensions to 0
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = 0;
        }
        // Allocate no memory
        this->total_size = 0;
        this->data = nullptr;
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 1), int>::type>
    Tensor<RANK>::Tensor( const size_t dim ) {
        if( dim == 0 ) {
            throw std::invalid_argument("The dimension size is less than 1.");
        }
        this->dimensions[0] = dim;
        this->total_size = dim;
        this->data = new float[dim];
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 1), int>::type>
    Tensor<RANK>::Tensor( const size_t dim, const float fill ) {
        if( dim == 0 ) {
            throw std::invalid_argument("The dimension size is less than 1.");
        }
        this->dimensions[0] = dim;
        this->total_size = dim;
        this->data = new float[dim];
        std::fill(this->data, this->data + this->total_size, fill);
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R > 1), int>::type>
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
        this->data = new float[total_size];
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R > 1), int>::type>
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
        this->data = new float[total_size];
        std::fill(this->data, this->data + this->total_size, fill);
    }
    template<size_t RANK>
    Tensor<RANK>::Tensor( InitializerElements<RANK> elements ) {
        if( elements.size() == 0 ) {
            throw std::invalid_argument("The first dimension size is less than 1.");
        }

        // Recursively count the total size of the initializer elements
        this->total_size = Tensor<RANK>::countInitializerElements(elements, this->dimensions);
        if( !Tensor<RANK>::checkInitializerElements(elements, this->dimensions) ) {
            throw std::invalid_argument("The given initializer elements are not rectangular");
        }

        // Check if any of the dimensions are 0
        for( size_t i = 0; i < RANK; ++i ) {
            if( this->dimensions[i] < 1 ) {
                throw std::invalid_argument("One or more dimension sizes are less than 1.");
            }
        }

        // Allocate memory
        this->data = new float[this->total_size];
        // Assign data from flattened initializer elements
        float* data_ptr = this->data;
        Tensor<RANK>::flattenInitializerElements(elements, data_ptr);
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R > 1), int>::type>
    Tensor<RANK>::Tensor( std::initializer_list<std::reference_wrapper<const VTensor<RANK-1>>> elements ) {
        const size_t dim1 = elements.size();
        if( dim1 == 0 ) {
            throw std::invalid_argument("The first dimension size is less than 1.");
        }
        
        const std::reference_wrapper<const VTensor<RANK-1>>* tensor_refs = elements.begin();
        // Copy dimensions from the first Tensor
        this->dimensions[0] = dim1;
        for( size_t i = 1; i < RANK; ++i ) {
            this->dimensions[i] = tensor_refs[0].get().dimensions[i-1];
        }
        // Check that all Tensors have the same dimensions
        for( size_t i = 1; i < dim1; ++i ) {
            for( size_t j = 0; j < RANK; ++j ) {
                if( this->dimensions[j] != tensor_refs[i].get().dimensions[j] ) {
                    throw std::invalid_argument("Two or more dimension sizes do not match.");
                }
            }
        }
        // Allocate memory for data
        const size_t inner_tensor_size = tensor_refs[0].get().total_size;
        this->total_size = dim1 * inner_tensor_size;
        this->data = new float[this->total_size];
        // Copy data from Tensors into this
        for( size_t i = 0; i < dim1; ++i ) {
            memcpy(this->data + (i * inner_tensor_size), tensor_refs[i].get().data, inner_tensor_size * sizeof(float));
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
        this->data = new float[other.total_size];
        std::memcpy( this->data, other.data, this->total_size * sizeof(float) );
    }
    template<size_t RANK>
    Tensor<RANK>::Tensor( const VTensor<RANK>& other ) {
        // Copy dimensions
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        // Allocate new memory and copy it over
        this->total_size = other.total_size;
        this->data = new float[other.total_size];
        std::memcpy( this->data, other.data, this->total_size * sizeof(float) );
    }
    template<size_t RANK>
    Tensor<RANK>::Tensor( Tensor<RANK>&& other ) {
        // Copy dimensions
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        // Set other's data pointer to this
        this->total_size = other.total_size;
        this->data = other.data;

        // Clear other tensor
        for( size_t i = 0; i < RANK; ++i ) {
            other.dimensions[i] = 0;
        }
        other.total_size = 0;
        other.data = nullptr;
    }
    template<size_t RANK>
    Tensor<RANK>::~Tensor() {
        delete this->data;
    }
    template<size_t RANK>
    Tensor<RANK>& Tensor<RANK>::operator = ( const Tensor<RANK>& other ) {
        // Check for self assignment
        if( this == &other ) {
            return *this;
        }
        // Free the previous data held in this Tensor.
        delete this->data;

        // Copy dimensions
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        // Allocate new memory and copy it over
        this->total_size = other.total_size;
        this->data = new float[other.total_size];
        std::memcpy( this->data, other.data, this->total_size * sizeof(float) );

        return *this;
    }
    template<size_t RANK>
    Tensor<RANK>& Tensor<RANK>::operator = ( const VTensor<RANK>& other ) {
        // Check for self assignment
        if( this == &other ) {
            return *this;
        }
        // Free the previous data held in this Tensor.
        delete this->data;

        // Copy dimensions
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        // Allocate new memory and copy it over
        this->total_size = other.total_size;
        this->data = new float[other.total_size];
        std::memcpy( this->data, other.data, this->total_size * sizeof(float) );

        return *this;
    }
    template<size_t RANK>
    Tensor<RANK>& Tensor<RANK>::operator = ( Tensor<RANK>&& other ) {
        // Free the previous data held in this Tensor.
        delete this->data;

        // Copy dimensions
        for( size_t i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        // Set other's data pointer to this
        this->total_size = other.total_size;
        this->data = other.data;

        // Clear other tensor
        for( size_t i = 0; i < RANK; ++i ) {
            other.dimensions[i] = 0;
        }
        other.total_size = 0;
        other.data = nullptr;
    }

    template<size_t RANK>
    size_t Tensor<RANK>::countInitializerElements( const InitializerElements<RANK>& elements, size_t dims[RANK] ) {
        dims[0] = elements.size();
        
        // RANK=1 case
        if constexpr( RANK == 1 ) {
            return elements.size();
        }

        // RANK>1 case
        else {
            const size_t inner_size = Tensor<RANK-1>::countInitializerElements(*elements.begin(), dims + 1);
            return elements.size() * inner_size;
        }
    }
    template<size_t RANK>
    bool Tensor<RANK>::checkInitializerElements( const InitializerElements<RANK>& elements, const size_t dims[RANK] ) {
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
                if( !Tensor<RANK-1>::checkInitializerElements(inner_elements, dims + 1) ) {
                    return false;
                }
            } 
            return true;
        }
    }
    template<size_t RANK>
    void Tensor<RANK>::flattenInitializerElements( const InitializerElements<RANK>& elements, float*& data ) {
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
                Tensor<RANK-1>::flattenInitializerElements(inner_elements, data);
            }
        }
    }
}



#endif