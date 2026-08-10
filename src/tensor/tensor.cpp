#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    bool contiguous = true;

    size_t ndim_ = this->shape().size();
    ptrdiff_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        if (this->strides()[ndim_ - i] != stride) {
            contiguous = false;
            break;
        }
        
        stride *= this->shape()[ndim_ - i];
    }

    return contiguous;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    size_t ndim_ = this->ndim();
    CHECK_ARGUMENT(order.size() == ndim_, "Tensor::permute: order size mismatch");

    std::vector<ptrdiff_t> strides(ndim_);
    std::vector<size_t> shape(ndim_);

    std::vector<bool> seen(ndim_, false);
    for (size_t i = 0; i < order.size(); i++) {
        CHECK_ARGUMENT(order[i] < ndim_, "Tensor::permute: order index out of bounds");
        CHECK_ARGUMENT(!seen[order[i]], "Tensor::permute: duplicate dimension");

        seen[order[i]] = true;
        strides[i] = this->strides()[order[i]];
        shape[i] = this->shape()[order[i]];
    }

    TensorMeta new_meta{this->dtype(), shape, strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    CHECK_ARGUMENT(this->numel() == new_numel, "Tensor::view: number of elements mismatch");

    auto new_meta = _meta;
    new_meta.shape = shape;
    std::vector<ptrdiff_t> strides(shape.size());

    std::vector<size_t> block_numels;
    std::vector<ptrdiff_t> block_base_strides;

    size_t block_numel = 1;
    ptrdiff_t block_base_stride = this->ndim() == 0 ? 1 : this->strides().back();
    for (size_t i = this->ndim(); i > 0; i--) {
        size_t dim = i - 1;
        block_numel *= this->shape()[dim];

        if (dim == 0 || this->strides()[dim - 1] != static_cast<ptrdiff_t>(this->shape()[dim]) * this->strides()[dim]) {
            block_numels.push_back(block_numel);
            block_base_strides.push_back(block_base_stride);
            block_numel = 1;
            if (dim > 0) {
                block_base_stride = this->strides()[dim - 1];
            }
        }
    }

    size_t new_dim_end = shape.size();
    for (size_t block = 0; block < block_numels.size(); block++) {
        size_t matched_numel = 1;
        size_t new_dim_begin = new_dim_end;

        while (new_dim_begin > 0 && matched_numel < block_numels[block]) {
            new_dim_begin--;
            matched_numel *= shape[new_dim_begin];
        }

        CHECK_ARGUMENT(matched_numel == block_numels[block], "Tensor::view: shape is incompatible with current strides");

        ptrdiff_t stride = block_base_strides[block];
        for (size_t i = new_dim_end; i > new_dim_begin; i--) {
            size_t dim = i - 1;
            strides[dim] = stride;
            stride *= shape[dim];
        }

        new_dim_end = new_dim_begin;
    }

    CHECK_ARGUMENT(new_dim_end == 0, "Tensor::view: shape is incompatible with current strides");

    new_meta.strides = strides;

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < this->ndim(), "Tensor::slice: dimension out of bounds");
    CHECK_ARGUMENT(start <= end && end <= this->shape()[dim], "Tensor::slice: invalid slice range");

    auto new_meta = _meta;
    new_meta.shape[dim] = end - start;

    size_t offset = _offset + start * this->strides()[dim] * this->elementSize();
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, offset));
}

void Tensor::load(const void *src_) {
    core::context().setDevice(this->deviceType(), this->deviceId());

    core::context().runtime().api()->memcpy_sync(
        this->data(),
        src_,
        this->numel() * this->elementSize(),
        LLAISYS_MEMCPY_H2D);
}

tensor_t Tensor::contiguous() const {
    if (this->isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }

    auto res = Tensor::create(
        this->shape(),
        this->dtype(),
        this->deviceType(),
        this->deviceId());

    size_t elem_size = this->elementSize();
    size_t total = this->numel();
    size_t ndim_ = this->ndim();

    for (size_t linear = 0; linear < total; linear++) {
        size_t tmp = linear;
        ptrdiff_t src_offset = 0;

        for (size_t rev = 0; rev < ndim_; rev++) {
            // 通过 linear 反代，得到新 tensor 中的 ij 下标
            // 有点类似进制转换
            // linear = i * (3*4) + j * 4 + k
            // 那么 k = linear % 4，然后求 ij 就继续 linear /= 4
            size_t dim = ndim_ - 1 - rev;

            size_t index = tmp % this->shape()[dim];
            tmp /= this->shape()[dim];

            // 获得原 tensor 中的偏移量，因为原 tensor 是非连续的，所以就一维一维地算 offset
            src_offset += index * this->strides()[dim];
        }

        std::memcpy(
            res->data() + linear * elem_size,
            this->data() + src_offset * elem_size,
            elem_size);
    }

    return res;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    try {
        return this->view(shape);
    } catch (...) {
        return this->contiguous()->view(shape);
    }
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    int target_device = device;
    if (target_device < 0) {
        target_device = device_type == this->deviceType() ? this->deviceId() : 0;
    }

    if (this->deviceType() == device_type && this->deviceId() == target_device) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }

    auto src = this->isContiguous()
        ? std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset))
        : this->contiguous();
    auto dst = Tensor::create(src->shape(), src->dtype(), device_type, target_device);

    llaisysMemcpyKind_t kind;
    if (src->deviceType() == LLAISYS_DEVICE_CPU && dst->deviceType() == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2H;
        core::context().setDevice(LLAISYS_DEVICE_CPU, 0);
    } else if (src->deviceType() == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2D;
        core::context().setDevice(dst->deviceType(), dst->deviceId());
    } else if (dst->deviceType() == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_D2H;
        core::context().setDevice(src->deviceType(), src->deviceId());
    } else {
        kind = LLAISYS_MEMCPY_D2D;
        core::context().setDevice(dst->deviceType(), dst->deviceId());
    }

    core::context().runtime().api()->memcpy_sync(
        dst->data(),
        src->data(),
        src->numel() * src->elementSize(),
        kind);

    return dst;
}

} // namespace llaisys
