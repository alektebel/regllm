# EFS filesystem for ChromaDB persistence across ECS task restarts
resource "aws_efs_file_system" "chroma" {
  encrypted        = true
  performance_mode = "generalPurpose"
  throughput_mode  = "bursting"

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }
}

resource "aws_efs_access_point" "chroma" {
  file_system_id = aws_efs_file_system.chroma.id

  posix_user {
    uid = 0
    gid = 0
  }

  root_directory {
    path = "/chroma"
    creation_info {
      owner_uid   = 0
      owner_gid   = 0
      permissions = "755"
    }
  }
}

resource "aws_efs_mount_target" "chroma" {
  for_each        = toset(var.subnet_ids)
  file_system_id  = aws_efs_file_system.chroma.id
  subnet_id       = each.value
  security_groups = [var.efs_security_group_id]
}
