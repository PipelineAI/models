pipeline {
  agent {
    dockerfile {
      filename 'Dockerfile'
    }
    
  }
  stages {
    stage('test') {
      steps {
        sh 'pip install cli-pipeline==1.4.31 --ignore-installed --no-cache -U'
      }
    }
  }
}