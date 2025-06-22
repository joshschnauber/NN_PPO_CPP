/** Tensor.hpp
 *  g++ -g -Wextra -Wall Tensor.hpp -o tensor_test.exe
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



// Declaration
namespace jai {
    /* Tensor Declarations */
    template<size_t RANK>
    class VTensor;
    template<size_t RANK>
    class Tensor;


    /* View Tensor
     * This represents the view into a part or whole of another Tensor.
     * Any instance of a VTensor, and the data contained within, is backed by a Tensor.
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

        /* We are not allowing Tensors to be casted to VTensors.
         * Important to remember that VTensors only provide a view of a Tensor;
         * the assigned VTensor is still dependent on the assigning Tensor.
         * If disallowing slicing from a Tensor to a VTensor is found to be a mistake,
         * comment the contructor and assignment operator below.
         */
        VTensor( const Tensor<RANK>& ) = delete;
        VTensor<RANK>& operator = ( Tensor<RANK>&& ) = delete;

        protected:

        VTensor( const size_t dimensions[RANK], size_t total_size, const float* data );

        /* Accessors */
        public:

        /* Defined for RANK=1 Tensors, this returns the element at the given index in the first (and only) dimension.
         */
        template<size_t R = RANK, typename std::enable_if<(R == 1), int>::type = 0>
        const float& operator [] ( size_t index ) const;
        /* Defined for RANK=1 Tensors, this returns a mutable reference to the element at the `index` in the first (and only) dimension.
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
        /* Returns an immutable View Tensor with rank RANK+1 of `this` Tensor, where it's last dimension is of size 1.
         * The returned View Tensor is backed by `this` Tensor.
         * Useful for converting a Vector into an (n x 1) Matrix for matrix multiplication.
         */
        const VTensor<RANK+1> rankUp() const;
        /* Returns an immutable View Tensor with rank 1 of `this` Tensor, all of it's values are flattened into one vector.
         * The returned View Tensor is backed by `this` Tensor.
         */
        const VTensor<1> flattened() const;

        /* Binary Operations */
        public:

        /* Adds all of the elements in the other Tensor to all of the elements in this Tensor and returns the result.
         * Both Tensors must be the same total size, but do not necessarily have to have the same dimensions.
         * The dimensions of this Tensor are passed onto the result Tensor.
         */
        Tensor<RANK> operator + ( const VTensor<RANK>& other ) const;
        /* Subtracts all of the elements in the other Tensor from all of the elements in this Tensor and returns the result.
         * Both Tensors must be the same total size, but do not necessarily have to have the same dimensions.
         * The dimensions of this Tensor are passed onto the result Tensor.
         */
        Tensor<RANK> operator - ( const VTensor<RANK>& other ) const;
        /* Multiplies all of the elements in `tensor` by `scale` and returns the result.
        */
        friend Tensor<RANK> operator * ( const VTensor<RANK>& tensor, float scale );
        /* Multiplies all of the elements in `tensor` by `scale` and returns the result.
        */
        friend Tensor<RANK> operator * ( float scale, const VTensor<RANK>& tensor );
        /* Divides all of the elements in `tensor` by `scale` and returns the result.
        */
        friend Tensor<RANK> operator / ( const VTensor<RANK>& tensor, float scale );
        /* Negates all of the elements in this Tensor and returns the result.
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

        /* Prints out the Tensor as a string.
         */
        friend std::ostream& operator << ( std::ostream& fs, const VTensor<RANK>& t );

        protected:
        size_t dimensions[RANK];
        size_t total_size;
        float* data;
    };


    /* Tensor
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
        /* Defined for RANK=1 Tensors, constructs a Tensor initialized with the given scalar elements.
         * The size of the Tensor is the size of `elements`.
         */
        template<size_t R = RANK, typename std::enable_if<(R == 1), int>::type = 0>
        Tensor( std::initializer_list<float> elements );
        /* Defined for RANK>1 Tensors, constructs a Tensor initialized with the given `Tensor<RANK-1>` elements.
         * The size of the first dimension is the size of `elements`.
         * Throws an error if any of the Tensors in `elements` have differing dimensions.
         */
        template<size_t R = RANK, typename std::enable_if<(R > 1), int>::type = 0>
        Tensor( std::initializer_list<std::reference_wrapper<const VTensor<RANK-1>>> elements );

        /* Copy constructor.
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
        Tensor<RANK>& operator = ( const VTensor<RANK>& other );
        /* Move assignment operator. Ensures that memory is freed when existing object is overwritten.
         */
        Tensor<RANK>& operator = ( Tensor<RANK>&& other );
    };


    // Vector and Matrix type definitions
    typedef VTensor<1> Vector;
    typedef VTensor<2> Matrix;
}



// Implementation
namespace jai {
    /* VTensor Implementation */

    template<size_t RANK>
    VTensor<RANK>::VTensor() {
        // Set all dimensions to 0
        for( int i = 0; i < RANK; ++i ) {
            this->dimensions[i] = 0;
        }
        // Allocate no memory
        this->total_size = 0;
        this->data = nullptr;
    }
    template<size_t RANK>
    VTensor<RANK>::VTensor( const VTensor<RANK>& other ) {
        for( int i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        this->total_size = other.total_size;
        this->data = other.data;
    }
    template<size_t RANK>
    VTensor<RANK>& VTensor<RANK>::operator = ( const VTensor<RANK>& other ) {
        for( int i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        this->total_size = other.total_size;
        this->data = other.data;

        return *this;
    }
    template<size_t RANK>
    VTensor<RANK>::VTensor( VTensor<RANK>&& other ) {
        for( int i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        this->total_size = other.total_size;
        this->data = other.data;
    }
    template<size_t RANK>
    VTensor<RANK>& VTensor<RANK>::operator = ( VTensor<RANK>&& other ) {
        for( int i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        this->total_size = other.total_size;
        this->data = other.data;

        return *this;
    }
    template<size_t RANK>
    VTensor<RANK>::VTensor( const size_t dimensions[RANK], const size_t total_size, const float* data ) {
        for( int i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        this->total_size = total_size;
        this->data = data;
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
        const size_t* dims = this->dimensions + 1;
        const size_t inner_tensor_total_size = this->total_size / this->dimensions[0];
        const float* inner_data = this->data + inner_tensor_total_size*index;

        return VTensor<RANK-1>(dims, inner_tensor_total_size, inner_data);
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R > 1), int>::type>
    VTensor<RANK-1> VTensor<RANK>::operator [] ( const size_t index ) {
        const size_t* dims = this->dimensions + 1;
        const size_t inner_tensor_total_size = this->total_size / this->dimensions[0];
        const float* inner_data = this->data + inner_tensor_total_size*index;

        return VTensor<RANK-1>(dims, inner_tensor_total_size, inner_data);
    }

    template<size_t RANK>
    const VTensor<RANK> VTensor<RANK>::view() const {
        return VTensor<RANK>(*this);
    }
    template<size_t RANK>
    const VTensor<RANK+1> VTensor<RANK>::rankUp() const {
        VTensor<RANK+1> rankedUpView;
        for( int i = 0; i < RANK; ++i ) {
            rankedUpView.dimensions[i] = this->dimensions[i];
        }
        rankedUpView.dimensions[RANK] = 1;
        rankedUpView.total_size = this->total_size;
        rankedUpView.data = this->data;
    }
    template<size_t RANK>
    const VTensor<1> VTensor<RANK>::flattened() const {
        return VTensor<1>({this->total_size}, this->total_size, this->data);
    }

    template<size_t RANK>
    Tensor<RANK> VTensor<RANK>::operator + ( const VTensor<RANK>& other ) const {
        // Copy this to new Tensor and addTo other to it
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
        result.mulBy(scale);
        // Return result Tensor
        return result;
    }
    template<size_t RANK>
    Tensor<RANK> operator * ( const float scale, const VTensor<RANK>& tensor ) {
        // Copy this to new Tensor and multiply it by scale
        Tensor<RANK> result(tensor);
        result.mulBy(scale);
        // Return result Tensor
        return result;
    }
    template<size_t RANK>
    Tensor<RANK> operator / ( const VTensor<RANK>& tensor, const float scale ) {
        // Copy this to new Tensor and divide it by scale
        Tensor<RANK> result(tensor);
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
        std::memcpy( this->data, tensor.data, this->total_size );
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
        return sqrt(this->squaredMag());
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 1), int>::type>
    float VTensor<RANK>::squaredMag() const {
        float sqrd_sum = 0;
        for( int i = 0; i < this->dimensions[0]; ++i ) {
            sqrd_sum += data[i] * data[i];
        }
        return sqrd_sum;
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 1), int>::type>
    float VTensor<RANK>::dot( const VTensor<1>& other ) const {
        float sum = 0;
        for( int i = 0; i < this->dimensions[0]; ++i ) {
            sum += this[i] * other[i]
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
        for( int i = 0; i < result.dimensions[0]; ++i ) {
            for( int j = 0; j < result.dimensions[1]; ++j ) {
                float sum = 0;
                for( int k = 0; k < this->dimensions[1]; ++k ) {
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
        for( int i = 0; i < result.dimensions[0]; ++i ) {
            float sum = 0;
            for( int j = 0; j < result.dimensions[1]; ++j ) {
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
                const float temp = this[{i, j}]
                this[{i, j}] = this[{j, i}]
                this[{j, i}] = temp
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
    std::ostream& operator << ( std::ostream& fs, const VTensor<RANK>& t ) {
        // Open Tensor
        fs << "{ ";
        // Print inner Tensors
        fs << t[0];
        for( int i = 1; i < t.dimensions[0]; ++i ) {
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
        for( int i = 0; i < RANK; ++i ) {
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
        this->data = new float[dim](fill);
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R > 1), int>::type>
    Tensor<RANK>::Tensor( const size_t (&dims)[RANK] ) {
        // Copy dimensions
        size_t total_size = 1;
        for( int i = 0; i < RANK; ++i ) {
            this->dimensions[i] = dims[i];
            total_size *= dims[i];
            // Check if the size of this dimension is 0
            if( dims[i] < 1 ) {
                throw std::invalid_argument("One or more dimension sizes are less than 1.");
            }
        }
        // Allocate memory for data
        this->total_size = total_size;
        data = new float[total_size];
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R > 1), int>::type>
    Tensor<RANK>::Tensor( const size_t (&dims)[RANK], const float fill ) {
        // Copy dimensions
        size_t total_size = 1;
        for( int i = 0; i < RANK; ++i ) {
            this->dimensions[i] = dims[i];
            total_size *= dims[i];
            // Check if the size of this dimension is 0
            if( dims[i] < 1 ) {
                throw std::invalid_argument("One or more dimension sizes are less than 1.");
            }
        }
        // Allocate memory for data and fill with value
        this->total_size = total_size;
        data = new float[total_size](fill);
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 1), int>::type>
    Tensor<RANK>::Tensor( std::initializer_list<float> elements ) : Tensor(elements.size()) {
        memcpy(this->data, elements.begin(), elements.size());
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R > 1), int>::type>
    Tensor<RANK>::Tensor( std::initializer_list<std::reference_wrapper<const VTensor<RANK-1>>> elements ) {
        const size_t dim1 = elements.size();
        const std::reference_wrapper<const Tensor<RANK>>* tensor_refs = elements.begin();
        // Copy dimensions from the first Tensor
        this->dimensions[0] = dim1;
        size_t total_size = 1;
        for( int i = 1; i < RANK; ++i ) {
            this->dimensions[i] = tensor_refs[0].get().dimensions[i-1];
            total_size *= tensor_refs[0].get().dimensions[i-1];
        }
        // Check that all Tensors have the same dimensions
        for( int i = 1; i < dim1; ++i ) {
            for( int j = 0; j < RANK; ++j ) {
                if( this->dimensions[j] != tensor_refs[i].get().dimensions[j] ) {
                    throw std::invalid_argument("Two or more dimension sizes do not match.");
                }
            }
        }
        // Allocate memory for data
        this->total_size = total_size;
        data = new float[total_size];
        // Copy data from Tensors into this
        const size_t inner_tensor_size = tensor_refs[0].get().total_size;
        for( int i = 0; i < dim1; ++i ) {
            memcpy(this->data + i * inner_tensor_size, tensor_refs[i].get().data, inner_tensor_size);
        }
    }

    template<size_t RANK>
    Tensor<RANK>::Tensor( const VTensor<RANK>& other ) {
        // Copy dimensions
        for( int i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        // Allocate new memory and copy it over
        this->total_size = other.total_size;
        this->data = new float[other.total_size];
        std::memcpy( this->data, other.data, this->total_size );
    }
    template<size_t RANK>
    Tensor<RANK>::Tensor( Tensor<RANK>&& other ) {
        // Copy dimensions
        for( int i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        // Set other's data pointer to this
        this->total_size = other.total_size;
        this->data = other.data;

        // Clear other tensor
        for( int i = 0; i < RANK; ++i ) {
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
    Tensor<RANK>& Tensor<RANK>::operator = ( const VTensor<RANK>& other ) {
        // Check for self assignment
        if( this == &other ) {
            return *this;
        }
        // Free the previous data held in this Tensor.
        delete this->data;

        // Copy dimensions
        for( int i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        // Allocate new memory and copy it over
        this->total_size = other.total_size;
        this->data = new float[other.total_size];
        std::memcpy( this->data, other.data, this->total_size );

        return *this;
    }
    template<size_t RANK>
    Tensor<RANK>& Tensor<RANK>::operator = ( Tensor<RANK>&& other ) {
        // Free the previous data held in this Tensor.
        delete this->data;

        // Copy dimensions
        for( int i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        // Set other's data pointer to this
        this->total_size = other.total_size;
        this->data = other.data;

        // Clear other tensor
        for( int i = 0; i < RANK; ++i ) {
            other.dimensions[i] = 0;
        }
        other.total_size = 0;
        other.data = nullptr;
    }
}



#endif TENSOR_HPP