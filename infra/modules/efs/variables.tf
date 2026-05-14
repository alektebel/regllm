variable "app_name"              { type = string }
variable "environment"           { type = string }
variable "efs_security_group_id" { type = string }
variable "subnet_ids"            { type = list(string) }
