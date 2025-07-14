provider "aws" {
  region = var.aws_region
}

resource "aws_instance" "flight_api" {
  ami           = "ami-0c2b8ca1dad447f8a" # Ubuntu 22.04 (verifica)
  instance_type = var.instance_type
  key_name      = var.key_name

  user_data = file("user_data.sh")

  tags = {
    Name = "flight-delay-api"
  }

  vpc_security_group_ids = [aws_security_group.api_sg.id]
}

resource "aws_security_group" "api_sg" {
  name        = "allow_api_access"
  description = "Allow port 8000"

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
