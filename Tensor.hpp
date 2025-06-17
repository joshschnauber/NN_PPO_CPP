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



// Declaration
namespace jai {
    
    /* Constant Tensor */
    template<size_t RANK>
    class _Tensor {
        int i;
    };

    template<size_t RANK>
    class CTensor : _Tensor<RANK> {

    };


    template<size_t RANK>
    class Tensor {
        // Ensure that Tensor RANK cannot be 0 (must have 1 or more dimensions)
        static_assert(RANK > 0, "Tensor rank cannot be 0.");

        public:
        /* Constructors */

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
         * The size of the first dimension is the size of `elements`.
         */
        template<size_t R = RANK, typename std::enable_if<(R == 1), int>::type = 0>
        Tensor( std::initializer_list<float> elements );
        /* Defined for RANK>1 Tensors, constructs a Tensor initialized with the given `Tensor<RANK-1>` elements.
         * The size of the first dimension is the size of `elements`.
         * Throws an error if any of the Tensors in `elements` have differing dimensions.
         */
        template<size_t R = RANK, typename std::enable_if<(R > 1), int>::type = 0>
        Tensor( std::initializer_list<Tensor<RANK-1>> elements );

        /* Copy constructor.
         */
        Tensor( const Tensor<RANK>& other );
        /* Move constructor.
         */
        Tensor( Tensor<RANK>&& other );
        /* Destructor.
         */
        ~Tensor();
        /* Assignment operator. Ensures that memory is freed when existing object is overwritten.
         */
        Tensor<RANK>& operator = ( const Tensor<RANK>& other );
        /* Move assignment operator. Ensures that memory is freed when existing object is overwritten.
         */
        Tensor<RANK>& operator = ( Tensor<RANK>&& other );
        
        /* Disallow casting from CTensor to Tensor
         */
        Tensor( const CTensor<RANK>& other ) = delete;

        /* Accessors */

        /* Defined for RANK=1 Tensors, this returns a mutable reference to the element at the given index in the first (and only) dimension.
         */
        template<size_t R = RANK, typename std::enable_if<(R == 1), int>::type = 0>
        float& operator [] ( size_t index );
        /* Defined for RANK=1 Tensors, this returns the element at the given index in the first (and only) dimension.
         */
        template<size_t R = RANK, typename std::enable_if<(R == 1), int>::type = 0>
        const float& operator [] ( size_t index ) const;

        /* Defined for RANK>1 Tensors, returns a mutable reference to the element at the given indexes.
         */
        template<size_t R = RANK, typename std::enable_if<(R > 1), int>::type = 0>
        float& operator [] ( const size_t (&indexes)[RANK] );
        /* Defined for RANK>1 Tensors, returns the element at the given indexes.
         */
        template<size_t R = RANK, typename std::enable_if<(R > 1), int>::type = 0>
        const float& operator [] ( const size_t (&indexes)[RANK] ) const;

        /* Defined for RANK>1 Tensors, this returns the Tensor of rank RANK-1 at the given index in the first dimension.
         */
        template<size_t R = RANK, typename std::enable_if<(R > 1), int>::type = 0>
        const Tensor<RANK-1> operator [] ( size_t index ) const;

        /* General operations */
        
        /* Adds all of the elements in the other Tensor to all of the elements in this Tensor.
         * The other Tensor must be the same total size as this Tensor, but does not necessarily have to have the same dimensions.
         */
        void addTo( const Tensor& other );
        /* Subtracts all of the elements in the other Tensor from all of the elements in this Tensor.
         * The other Tensor must be the same total size as this Tensor, but does not necessarily have to have the same dimensions.
         */
        void subFrom( const Tensor& other );
        /* Multiples all of the elements in this Tensor with the given scale.
         */
        void scaleBy( float scale );

        /* Adds all of the elements in the other Tensor to all of the elements in this Tensor and returns the result.
         * Both Tensors must be the same total size, but do not necessarily have to have the same dimensions.
         * The dimensions of this Tensor are passed onto the result Tensor.
         */
        Tensor<RANK> operator + ( const Tensor<RANK>& other ) const;
        /* Subtracts all of the elements in the other Tensor from all of the elements in this Tensor and returns the result.
         * Both Tensors must be the same total size, but do not necessarily have to have the same dimensions.
         * The dimensions of this Tensor are passed onto the result Tensor.
         */
        Tensor<RANK> operator - ( const Tensor<RANK>& other ) const;
        /* Multiplies all of the elements in this Tensor with the given scale and returns the result.
         */
        Tensor<RANK> operator * ( float scale ) const;
        /* Divides all of the elements in this Tensor with the given scale and returns the result.
         */
        Tensor<RANK> operator / ( float scale ) const;
        /* Negates all of the elements in this Tensor and returns the result.
         */
        Tensor<RANK> operator - () const;
        /* Returns this Tensor with a rank of RANK+1, where it's last dimension is of size 1.
         * Useful for converting a Vector into a Matrix for matrix multiplication.
         */
        Tensor<RANK+1> rankUp() const;

        /* Vector and Matrix operations */

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
        float dot( const Tensor<1>& other ) const;
        /* Takes the cross product of this Vector with the other Vector and returns the result.
         * The two vectors must be 3 dimensional.
         */
        template<size_t R = RANK, typename std::enable_if<(R == 1), int>::type = 0>
        Tensor<1> cross( const Tensor<1>& other ) const;

        /* Transposes this Matrix.
         */
        template<size_t R = RANK, typename std::enable_if<(R == 2), int>::type = 0>
        void transpose();
        /* Finds the matrix multiplication of the other Matrix on this Matrix and returns the result.
         * This Matrix must be of size (m x n) and the other Matrix must be of size (n x w)
         */
        template<size_t R = RANK, typename std::enable_if<(R == 2), int>::type = 0>
        Tensor<2> mul( const Tensor<2>& other ) const;

        /* Getters */

        /* Returns the rank of the tensor (the number of dimensions).
         * NOTE: This is NOT the matrix rank.
         */
        size_t rank() const;
        /* Returns the size of the Tensor in the given dimension.
         */
        size_t size( size_t dimension ) const;
        /* Returns the total size of the Tensor (the total number of elements).
         */
        size_t totalSize() const;

        /* Prints out the Tensor as a string.
         */
        friend std::ostream& operator << ( std::ostream& fs, const Tensor<RANK>& t );


        private:
        size_t dimensions[RANK];
        size_t total_size;
        float* data;
    };

    typedef Tensor<1> Vector;
    typedef Tensor<2> Matrix;


    /* Constant Copy Tensor */
    template<size_t RANK>
    class CTensor : public Tensor<RANK> {
        public:
        CTensor( size_t dimensions[RANK], size_t total_size, float* data ) {
            this->dimensions = dimensions;
            this->total_size = total_size;
            this->data = data;
        }
        CTensor( const CTensor& other) {
            
        }
    };
}



// Implementation
namespace jai {

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
    Tensor<RANK>::Tensor( std::initializer_list<Tensor<RANK-1>> elements ) {
        const size_t dim1 = elements.size();
        const Tensor<RANK>* tensors = elements.begin();
        // Copy dimensions from the first Tensor
        this->dimensions[0] = dim1;
        size_t total_size = 1;
        for( int i = 1; i < RANK; ++i ) {
            this->dimensions[i] = tensors->dimensions[i-1];
            total_size *= tensors->dimensions[i-1];
        }
        // Check that all Tensors have the same dimensions
        for( int i = 1; i < dim1; ++i ) {
            for( int j = 0; j < RANK; ++j ) {
                if( this->dimensions[j] != tensors[i].dimensions[j] ) {
                    throw std::invalid_argument("Two or more dimension sizes do not match.");
                }
            }
        }
        // Allocate memory for data
        this->total_size = total_size;
        data = new float[total_size];
        // Copy data from Tensors into this
        const size_t inner_tensor_size = tensors->total_size;
        for( int i = 0; i < dim1; ++i ) {
            memcpy(this->data + i * inner_tensor_size, tensors[i].data, inner_tensor_size);
        }
    }
    
    template<size_t RANK>
    Tensor<RANK>::Tensor( const Tensor<RANK>& other ) {
        // Copy dimensions
        for( int i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        // Allocate new memory and copy it over
        this->total_size = other.total_size;
        this->data = new float[other.total_size];
        std::memcpy( this->data, other.data );
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
    Tensor<RANK>& Tensor<RANK>::operator = ( const Tensor<RANK>& other ) {
        // Free the previous data held in this Tensor.
        delete this->data;

        // Copy dimensions
        for( int i = 0; i < RANK; ++i ) {
            this->dimensions[i] = other.dimensions[i];
        }
        // Allocate new memory and copy it over
        this->total_size = other.total_size;
        this->data = new float[other.total_size];
        std::memcpy( this->data, other.data );
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

    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 1), int>::type>
    float& Tensor<RANK>::operator [] ( const size_t index ) {
        return this->data[index];
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 1), int>::type>
    const float& Tensor<RANK>::operator [] ( const size_t index ) const {
        return this->data[index];
    }
    
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R > 1), int>::type>
    float& Tensor<RANK>::operator [] ( const size_t (&indexes)[RANK] ) {
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
    const float& Tensor<RANK>::operator [] ( const size_t (&indexes)[RANK] ) const {
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
    const Tensor<RANK-1> Tensor<RANK>::operator [] ( const size_t index ) const {
        // Create inner Tensor
        Tensor<RANK-1> inner_tensor(this->dimensions + 1);
        const size_t inner_tensor_total_size = this->total_size / this->dimensions[0];
        // Set values in inner Tensor
        for( int i = 0; i < inner_tensor_total_size; ++i ) {
            inner_tensor.data[i] = this->data[inner_tensor_total_size*index + i];
        }
        return inner_tensor;
    }
    

    template<size_t RANK>
    void Tensor<RANK>::addTo( const Tensor& other ) {
        // Add other's values
        for( size_t i = 0; i < this->total_size; ++i ) {
            this->data[i] += other.data[i];
        }
    }
    template<size_t RANK>
    void Tensor<RANK>::subFrom( const Tensor& other ) {
        // Subtract other's values
        for( size_t i = 0; i < this->total_size; ++i ) {
            this->data[i] -= other.data[i];
        } 
    }
    template<size_t RANK>
    void Tensor<RANK>::scaleBy( const float scale ) {
        // Multiply by scale
        for( size_t i = 0; i < this->total_size; ++i ) {
            this->data[i] *= scale;
        }
    }

    template<size_t RANK>
    Tensor<RANK> Tensor<RANK>::operator + ( const Tensor<RANK>& other ) const {
        // Copy this to new Tensor and addTo other to it
        Tensor<RANK> result(*this);
        result.addTo(other);
        // Return result Tensor
        return result;
    }
    template<size_t RANK>
    Tensor<RANK> Tensor<RANK>::operator - ( const Tensor<RANK>& other ) const {
        // Copy this to new Tensor and subtract other from it
        Tensor<RANK> result(*this);
        result.subFrom(other);
        // Return result Tensor
        return result;
    }
    template<size_t RANK>
    Tensor<RANK> Tensor<RANK>::operator * ( float scale ) const {
        // Copy this to new Tensor and multiply it by scale
        Tensor<RANK> result(*this);
        result.mulBy(scale);
        // Return result Tensor
        return result;
    }
    template<size_t RANK>
    Tensor<RANK> Tensor<RANK>::operator / ( float scale ) const {
        // Copy this to new Tensor and divide it by scale
        Tensor<RANK> result(*this);
        result.scaleBy(1.0f / scale);
        // Return result Tensor
        return result;
    }
    template<size_t RANK>
    Tensor<RANK> Tensor<RANK>::operator - () const {
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
    Tensor<RANK+1> Tensor<RANK>::rankUp() const {
        // Create new Tensor with same dimensions as this Tensor
        size_t new_dimensions[RANK+1];
        for( int i = 0; i < RANK; ++i ) {
            new_dimensions[i] = this->dimensions[i];
        }
        new_dimensions[RANK] = 1;
        Tensor<RANK+1> ranked_up_tensor(new_dimensions);

        // Copy data into new tensor (the total size should be the same)
        for( int i = 0; i < this->total_size; ++i ) {
            ranked_up_tensor.data[i] = this->data[i];
        }

        return ranked_up_tensor;
    }

    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 1), int>::type>
    float Tensor<RANK>::mag() const {
        return sqrt(this->squaredMag());
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 1), int>::type>
    float Tensor<RANK>::squaredMag() const {
        float sqrd_sum = 0;
        for( int i = 0; i < this->dimensions[0]; ++i ) {
            sqrd_sum += data[i] * data[i];
        }
        return sqrd_sum;
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 1), int>::type>
    float Tensor<RANK>::dot( const Tensor<1>& other ) const {
        float sum = 0;
        for( int i = 0; i < this->dimensions[0]; ++i ) {
            sum += this[i] * other[i]
        }
        return sum;
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 1), int>::type>
    Tensor<1> Tensor<RANK>::cross( const Tensor<1>& other ) const {
        Tensor<1> result(3);
        result[0] = this[1] * other[2] - this[2] * other[1];
        result[1] = this[2] * other[0] - this[0] * other[2];
        result[2] = this[0] * other[1] - this[1] * other[0];
        return result;
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 2), int>::type>
    void Tensor<RANK>::transpose() {
        for( int i = 0; i < this->dimensions[0]; ++i ) {
            for( int j = i + 1; j < this->dimensions[1]; ++j ) {
                const float temp = this[{i, j}]
                this[{i, j}] = this[{j, i}]
                this[{j, i}] = temp
            }
        }
    }
    template<size_t RANK>
    template<size_t R, typename std::enable_if<(R == 2), int>::type>
    Tensor<2> Tensor<RANK>::mul( const Tensor<2>& other ) const {
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
    size_t Tensor<RANK>::rank() const {
        return RANK;
    }
    template<size_t RANK>
    size_t Tensor<RANK>::size( const size_t dimension ) const {
        return this->dimensions[dimension];
    }
    template<size_t RANK>
    size_t Tensor<RANK>::totalSize() const {
        return this->total_size;
    }

    template<size_t RANK>
    std::ostream& operator << ( std::ostream& fs, const Tensor<RANK>& t ) {
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

}



/* Unit tests. 
 */
int main() {
    jai::Tensor<1> t1 = jai::Tensor<1>(5, 0);
    t1[1];
    t1[{1}];

    jai::Tensor<2> t2 = jai::Tensor<2>({3,2}, 0);
    t2[1];
    t2[{1,2}];

    jai::Tensor<3> t3 = jai::Tensor<3>({2, 3, 2}, 0);

    //jai::CTensor<2> ct = t3[0];
    //ct.scaleBy(1);
    //jai::Tensor<2> t = ct;
    jai::Tensor<2> t = t3[0];
}



#endif TENSOR_HPP