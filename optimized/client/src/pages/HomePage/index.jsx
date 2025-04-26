import React from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Box, 
  Typography, 
  Button, 
  Grid, 
  Card, 
  CardContent, 
  CardActions,
  CardMedia,
  Container,
  Paper
} from '@mui/material';
import { 
  Security as SecurityIcon,
  School as SchoolIcon,
  Dashboard as DashboardIcon,
  Article as ArticleIcon
} from '@mui/icons-material';

const features = [
  {
    title: 'Cybersecurity Training',
    description: 'Access comprehensive cybersecurity courses designed for all skill levels, from beginners to experts.',
    icon: <SchoolIcon fontSize="large" color="primary" />,
    action: '/courses',
    actionText: 'Browse Courses'
  },
  {
    title: 'Security Scanner',
    description: 'Scan files and code for potential security threats using our AI-powered security scanner.',
    icon: <SecurityIcon fontSize="large" color="primary" />,
    action: '/security-scanner',
    actionText: 'Try Scanner'
  },
  {
    title: 'Security Dashboard',
    description: 'Monitor your security status and track your learning progress in one central dashboard.',
    icon: <DashboardIcon fontSize="large" color="primary" />,
    action: '/dashboard',
    actionText: 'View Dashboard'
  },
  {
    title: 'Learning Resources',
    description: 'Access a library of cybersecurity resources, articles, and best practices.',
    icon: <ArticleIcon fontSize="large" color="primary" />,
    action: '/resources',
    actionText: 'View Resources'
  }
];

const HomePage = () => {
  const navigate = useNavigate();

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Hero Section */}
      <Paper 
        sx={{
          position: 'relative',
          backgroundColor: 'rgba(0,0,0,.5)',
          color: '#fff',
          mb: 4,
          backgroundSize: 'cover',
          backgroundRepeat: 'no-repeat',
          backgroundPosition: 'center',
          backgroundImage: 'url(/images/hero-background.jpg)',
          padding: '4rem 0'
        }}
      >
        <Container maxWidth="md">
          <Box 
            sx={{
              position: 'relative',
              padding: { xs: '2rem', md: '4rem' },
              backdropFilter: 'blur(10px)',
              backgroundColor: 'rgba(0,0,0,.3)',
              borderRadius: '8px'
            }}
          >
            <Typography 
              component="h1" 
              variant="h2" 
              color="inherit" 
              gutterBottom
            >
              EGen Security AI
            </Typography>
            <Typography 
              variant="h5" 
              color="inherit" 
              paragraph
            >
              Learn cybersecurity with AI-powered educational tools and security scanning capabilities
            </Typography>
            <Button 
              variant="contained" 
              size="large" 
              onClick={() => navigate('/courses')}
              sx={{ mt: 2 }}
            >
              Get Started
            </Button>
          </Box>
        </Container>
      </Paper>

      {/* Features Section */}
      <Container maxWidth="lg" sx={{ mb: 6 }}>
        <Typography 
          component="h2" 
          variant="h3" 
          align="center" 
          gutterBottom
        >
          Features
        </Typography>
        <Typography 
          variant="h6" 
          align="center" 
          color="textSecondary" 
          paragraph
          sx={{ mb: 6 }}
        >
          Explore the powerful features of EGen Security AI
        </Typography>

        <Grid container spacing={4}>
          {features.map((feature, index) => (
            <Grid item key={index} xs={12} sm={6} md={3}>
              <Card 
                sx={{ 
                  height: '100%', 
                  display: 'flex', 
                  flexDirection: 'column',
                  '&:hover': { 
                    transform: 'translateY(-4px)', 
                    boxShadow: 4,
                    transition: 'all 0.3s ease-in-out'
                  }
                }}
              >
                <Box 
                  sx={{ 
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    p: 2
                  }}
                >
                  {feature.icon}
                </Box>
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography gutterBottom variant="h5" component="h3" align="center">
                    {feature.title}
                  </Typography>
                  <Typography align="center">
                    {feature.description}
                  </Typography>
                </CardContent>
                <CardActions sx={{ justifyContent: 'center', pb: 2 }}>
                  <Button 
                    size="small" 
                    color="primary" 
                    onClick={() => navigate(feature.action)}
                  >
                    {feature.actionText}
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>

      {/* About Section */}
      <Container maxWidth="lg" sx={{ mb: 6 }}>
        <Typography 
          component="h2" 
          variant="h3" 
          align="center" 
          gutterBottom
        >
          About EGen Security AI
        </Typography>
        <Typography paragraph>
          EGen Security AI is a comprehensive cybersecurity education and security scanning platform designed to help individuals and organizations improve their security posture. Our AI-powered tools provide interactive learning experiences and powerful security scanning capabilities.
        </Typography>
        <Typography paragraph>
          Whether you're a beginner learning about cybersecurity fundamentals or an experienced professional looking to expand your knowledge, EGen Security AI provides the resources and tools you need to succeed.
        </Typography>
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
          <Button 
            variant="outlined" 
            size="large" 
            onClick={() => navigate('/about')}
          >
            Learn More
          </Button>
        </Box>
      </Container>
    </Box>
  );
};

export default HomePage; 