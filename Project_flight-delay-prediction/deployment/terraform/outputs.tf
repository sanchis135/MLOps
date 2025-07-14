output "public_ip" {
  value = aws_instance.flight_api.public_ip
  description = "Public IP of EC2 instance"
}
