# Mạng Nơ-ron tích chập (Convolutional Neural Networks - ConvNets) thường được phát triển với ngân sách tài nguyên cố định
# và sau đó được thu phóng để có độ chính xác tốt hơn nếu có nhiều tài nguyên hơn
# EfficientNet
#cân bằng một cách có hệ thống độ sâu, chiều rộng và độ phân giải mạng (network depth, width, and resolution)
# Depth là độ sâu của mạng tương đương với số lớp trong đó.
# Width là độ rộng của mạng. Ví dụ: một thước đo chiều rộng là số kênh trong lớp Conv
# Resolution là độ phân giải hình ảnh được chuyển đến CNN.

#Thu phóng theo chiều sâu là một cách thông dụng nhất được sử dụng để thu phóng một mô hình CNN.
# Độ sâu có thể được thu phóng cũng như thu nhỏ bằng cách thêm hoặc bớt các lớp tương ứng

#Đúng là có một số lý do mà việc thêm nhiều lớp ẩn hơn sẽ cung cấp mức độ chính xác hơn cho mô hình.
# Tuy nhiên, điều này chỉ đúng với các tập dữ liệu lớn hơn,
# vì càng nhiều lớp với hệ số bước ngắn hơn sẽ trích xuất nhiều tính năng hơn cho dữ liệu đầu vào của bạn

#Việc thu phóng theo chiều rộng của mạng (theo như trong hình minh họa ta có thể hiểu là thêm dữ liệu đầu vào)
# cho phép các lớp tìm hiểu các tính năng chi tiết hơn.
#Tuy nhiên, cũng như trường hợp tăng chiều sâu, tăng chiều rộng ngăn cản mạng học các tính năng phức tạp,
# dẫn đến giảm độ chính xác.

#Độ phân giải đầu vào cao hơn cung cấp hình ảnh chi tiết hơn
# và do đó nâng cao khả năng suy luận của mô hình về các đối tượng nhỏ hơn và trích xuất các mẫu mịn hơn

#Kết luận: Thu phóng quy mô bất kỳ kích thước nào về chiều rộng, chiều sâu hoặc độ phân giải của mạng
# sẽ cải thiện độ chính xác, nhưng độ chính xác sẽ giảm đối với các mô hình lớn hơn.

#https://viblo.asia/p/efficientnet-cach-tiep-can-moi-ve-model-scaling-cho-convolutional-neural-networks-Qbq5QQzm5D8

