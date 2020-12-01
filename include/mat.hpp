#pragma once
#include <vector>

namespace ffnn
{
	// Type helper
	template<typename T>
	using Vector = std::vector<T>;
	using Vectorf = Vector<float>;
	
	template<typename T>
	using Function = T (*)(T);

	template<typename T>
	struct Matrix
	{
		uint64_t width;
		uint64_t height;
		std::vector<T> values;

		Matrix()
			: width(0)
			, height(0)
			, values(0)
		{}

		Matrix(uint64_t w, uint64_t h)
			: width(w)
			, height(h)
			, values(w* h)
		{
		}

		Matrix(uint64_t w, uint64_t h, const Vector<T>& data)
			: width(w)
			, height(h)
			, values(data)
		{
		}

		Matrix<T> transpose() const
		{
			Matrix<T> result(height, width);
			for (uint64_t i(0); i < width; ++i) {
				for (uint64_t j(0); j < height; ++j) {
					result(j, i) = this->operator()(i, j);
				}
			}
			return result;
		}

		T& operator()(uint64_t column, uint64_t row)
		{
			return values[column + row * width];
		}

		const T& operator()(uint64_t column, uint64_t row) const
		{
			return values[column + row * width];
		}
	};

	template<typename T>
	Matrix<T> operator* (const Matrix<T>& m1, const Matrix<T>& m2)
	{
		Matrix<T> result(m1.height, m2.width);
		for (uint64_t col(0); col < result.width; ++col) {
			for (uint64_t row(0); row < result.height; ++row) {
				float sum = 0.0f;
				const uint64_t n = m1.width;
				for (uint64_t i(0); i < n; ++i) {
					sum += m1(i, row) * m2(col, i);
				}
				result(col, row) = sum;
			}
		}

		return result;
	}

	template<typename T>
	Matrix<T> asColumn(const Vector<T>& v)
	{
		return Matrix<T>(1, v.size(), v);
	}

	template<typename T>
	Matrix<T> transpose(const Vector<T>& v)
	{
		return Matrix<T>(v.size(), 1, v);
	}

	template<typename T>
	Matrix<T> operator+ (const Matrix<T>& m, T value)
	{
		Matrix<T> result(m);
		for (T& v : result.values) {
			v += value;
		}

		return result;
	}

	template<typename T>
	Matrix<T> operator+ (const Matrix<T>& m1, const Matrix<T>& m2)
	{
		Matrix<T> result(m1);
		uint64_t index = 0;
		for (T& v : result.values) {
			v += m2.values[index];
			++index;
		}

		return result;
	}

	template<typename T>
	Matrix<T> operator- (const Matrix<T>& m1, const Matrix<T>& m2)
	{
		Matrix<T> result(m1);
		uint64_t index = 0;
		for (T& v : result.values) {
			v -= m2.values[index];
			++index;
		}

		return result;
	}

	template<typename T>
	Matrix<T> operator- (const Matrix<T>& m, T value)
	{
		Matrix<T> result(m);
		for (T& v : result.values) {
			v -= value;
		}

		return result;
	}

	template<typename T>
	Matrix<T> operator* (const Vector<T>& v, const Matrix<T>& m)
	{
		Matrixf result(m.width, v.size());
		for (uint64_t col(0); col < result.width; ++col) {
			for (uint64_t row(0); row < result.height; ++row) {
				result(col, row) = v[row] * m.values[col];
			}
		}

		return result;
	}

	template<typename T>
	Matrix<T> operator* (const Matrix<T>& m, T value)
	{
		Matrix<T> result(m);
		for (T& v : result.values) {
			v *= value;
		}

		return result;
	}

	template<typename T>
	Matrix<T> operator* (T value, const Matrix<T>& m)
	{
		return m * value;
	}

	template<typename T>
	Vector<T> operator+ (const Vector<T>& v1, const Vector<T>& v2)
	{
		const uint64_t size = v2.size();
		Vector<T> result(size);
		for (uint64_t i(0); i < size; ++i) {
			result[i] = v1[i] + v2[i];
		}

		return result;
	}

	template<typename T>
	Vector<T> operator- (const Vector<T>& v1, const Vector<T>& v2)
	{
		const uint64_t size = v2.size();
		Vector<T> result(size);
		for (uint64_t i(0); i < size; ++i) {
			result[i] = v1[i] - v2[i];
		}

		return result;
	}

	template<typename T>
	Vector<T> operator* (const Vector<T>& v1, const Vector<T>& v2)
	{
		const uint64_t size = v2.size();
		Vector<T> result(size);
		for (uint64_t i(0); i < size; ++i) {
			result[i] = v1[i] * v2[i];
		}

		return result;
	}

	template<typename T>
	Vector<T> operator* (const Vector<T>& v, T f)
	{
		Vector<T> result(v);
		for (T& val : result) {
			val *= f;
		}

		return result;
	}

	template<typename T>
	Vector<T> operator- (T f, const Vector<T>& v)
	{
		Vector<T> result(v);
		for (T& val : result) {
			val = f - val;
		}

		return result;
	}

	template<typename T>
	Vector<T> operator* (T f, const Vector<T>& v)
	{
		return v * f;
	}

	template<typename T>
	Vector<T> operator* (const Matrix<T>& m, const Vector<T>& v)
	{
		Vector<T> result(m.height);
		for (uint64_t row(0); row < m.height; ++row) {
			float sum = 0.0f;
			for (uint64_t col(0); col < m.width; ++col) {
				sum += v[col] * m(col, row);
			}
			result[row] = sum;
		}

		return result;
	}

	template<typename T>
	Vector<T> map(Function<T> f, const Vector<T>& v)
	{
		const uint64_t size = v.size();
		Vector<T> result(size);
		for (uint64_t i(0); i<size; ++i) {
			result[i] = f(v[i]);
		}

		return result;
	}
	
	template<typename T>
	T dot(const Vector<T>& v1, const Vector<T>& v2)
	{
		T result{};
		const uint64_t dim = v1.size();
		for (uint64_t i(0); i<dim; ++i) {
			result += v1[i] * v2[i];
		}

		return result;
	}

	using Matrixf = Matrix<float>;

	// Get the ith element from the back
	template<typename T>
	const T& crget(const std::vector<T>& v, uint64_t i = 0)
	{
		return *(v.end() - (i + 1));
	}

	template<typename T>
	T& rget(std::vector<T>& v, uint64_t i = 0)
	{
		return const_cast<T&>(crget(v, i));
	}
}