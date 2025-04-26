import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Box, 
  Typography, 
  Container, 
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  Tabs,
  Tab,
  TextField,
  InputAdornment,
  CircularProgress,
  Pagination, 
  Divider
} from '@mui/material';
import { 
  Search as SearchIcon,
  School as SchoolIcon, 
  Security as SecurityIcon,
  Code as CodeIcon,
  DataObject as DataObjectIcon
} from '@mui/icons-material';
import { useDispatch, useSelector } from 'react-redux';

// Import API hooks
import { useGetCoursesQuery } from '../../store/api';

const categoryIcons = {
  'basics': <SchoolIcon color="primary" />,
  'intermediate': <SecurityIcon color="primary" />,
  'advanced': <CodeIcon color="primary" />,
  'expert': <DataObjectIcon color="primary" />
};

const categoryColors = {
  'basics': 'primary',
  'intermediate': 'success',
  'advanced': 'warning',
  'expert': 'error'
};

const CourseCard = ({ course, onClick }) => {
  const categoryColor = categoryColors[course.category] || 'default';
  const categoryIcon = categoryIcons[course.category] || <SchoolIcon />;
  
  return (
    <Card 
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        transition: 'transform 0.2s',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: 6
        }
      }}
    >
      <CardContent sx={{ flexGrow: 1, pb: 0 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
          <Chip 
            icon={categoryIcon}
            label={course.category} 
            size="small" 
            color={categoryColor}
            sx={{ textTransform: 'capitalize' }}
          />
          <Typography variant="caption" color="text.secondary">
            {course.estimated_time} mins
          </Typography>
        </Box>
        <Typography gutterBottom variant="h5" component="h2">
          {course.title}
        </Typography>
        <Typography color="text.secondary" paragraph>
          {course.description}
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
          {course.tags.map((tag, index) => (
            <Chip 
              key={index} 
              label={tag} 
              size="small" 
              variant="outlined" 
            />
          ))}
        </Box>
      </CardContent>
      <CardActions>
        <Button 
          size="small" 
          color="primary" 
          onClick={onClick}
        >
          Start Learning
        </Button>
      </CardActions>
    </Card>
  );
};

const CoursesPage = () => {
  const navigate = useNavigate();
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [page, setPage] = useState(1);
  const pageSize = 8;
  
  // API query with optional category filter
  const queryParams = selectedCategory === 'all' 
    ? { page, page_size: pageSize } 
    : { category: selectedCategory, page, page_size: pageSize };
  
  const { data, error, isLoading } = useGetCoursesQuery(queryParams);
  
  const handleCategoryChange = (event, newValue) => {
    setSelectedCategory(newValue);
    setPage(1); // Reset to first page when changing category
  };
  
  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value);
    setPage(1); // Reset to first page when searching
  };
  
  const handlePageChange = (event, value) => {
    setPage(value);
  };
  
  const handleCourseClick = (course) => {
    navigate(`/courses/${course.category}/${course.id}`);
  };
  
  // Filter courses by search term (client-side filtering)
  const filteredCourses = data?.courses.filter(course => 
    searchTerm === '' || 
    course.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    course.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
    course.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))
  ) || [];
  
  return (
    <Container maxWidth="lg">
      <Typography 
        component="h1" 
        variant="h3" 
        align="center" 
        gutterBottom
      >
        Cybersecurity Courses
      </Typography>
      <Typography 
        variant="h6" 
        align="center" 
        color="textSecondary" 
        paragraph
        sx={{ mb: 4 }}
      >
        Explore our comprehensive collection of cybersecurity courses for all skill levels
      </Typography>
      
      {/* Search and Filters */}
      <Box sx={{ mb: 4 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={8}>
            <TextField
              fullWidth
              placeholder="Search courses by title, description, or tags"
              value={searchTerm}
              onChange={handleSearchChange}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                ),
              }}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <Tabs
              value={selectedCategory}
              onChange={handleCategoryChange}
              variant="scrollable"
              scrollButtons="auto"
              sx={{ borderBottom: 1, borderColor: 'divider' }}
            >
              <Tab label="All" value="all" />
              <Tab label="Basics" value="basics" />
              <Tab label="Intermediate" value="intermediate" />
              <Tab label="Advanced" value="advanced" />
              <Tab label="Expert" value="expert" />
            </Tabs>
          </Grid>
        </Grid>
      </Box>
      
      {/* Courses Grid */}
      <Box sx={{ mb: 4 }}>
        {isLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Typography color="error" align="center">
            Error loading courses. Please try again later.
          </Typography>
        ) : filteredCourses.length === 0 ? (
          <Typography align="center" sx={{ py: 4 }}>
            No courses found matching your criteria.
          </Typography>
        ) : (
          <Grid container spacing={4}>
            {filteredCourses.map((course) => (
              <Grid item key={`${course.category}-${course.id}`} xs={12} sm={6} md={3}>
                <CourseCard
                  course={course}
                  onClick={() => handleCourseClick(course)}
                />
              </Grid>
            ))}
          </Grid>
        )}
      </Box>
      
      {/* Pagination */}
      {data && data.total > pageSize && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
          <Pagination 
            count={Math.ceil(data.total / pageSize)} 
            page={page} 
            onChange={handlePageChange} 
            color="primary" 
          />
        </Box>
      )}
      
      {/* Categories Overview */}
      <Box sx={{ mt: 6 }}>
        <Typography variant="h4" gutterBottom>
          Course Categories
        </Typography>
        <Divider sx={{ mb: 3 }} />
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <SchoolIcon color="primary" sx={{ mr: 1 }} />
                  <Typography variant="h5">
                    Basics
                  </Typography>
                </Box>
                <Typography variant="body2">
                  Introductory courses for beginners with no prior knowledge of cybersecurity. Learn fundamental concepts and build a strong foundation.
                </Typography>
              </CardContent>
              <CardActions>
                <Button size="small" onClick={() => setSelectedCategory('basics')}>
                  View Basics Courses
                </Button>
              </CardActions>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <SecurityIcon color="success" sx={{ mr: 1 }} />
                  <Typography variant="h5">
                    Intermediate
                  </Typography>
                </Box>
                <Typography variant="body2">
                  Courses for learners with basic knowledge who want to deepen their understanding of specific cybersecurity topics and practical skills.
                </Typography>
              </CardContent>
              <CardActions>
                <Button size="small" onClick={() => setSelectedCategory('intermediate')}>
                  View Intermediate Courses
                </Button>
              </CardActions>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <CodeIcon color="warning" sx={{ mr: 1 }} />
                  <Typography variant="h5">
                    Advanced
                  </Typography>
                </Box>
                <Typography variant="body2">
                  Advanced topics for experienced learners. These courses cover complex security concepts, tools, and techniques used by professionals.
                </Typography>
              </CardContent>
              <CardActions>
                <Button size="small" onClick={() => setSelectedCategory('advanced')}>
                  View Advanced Courses
                </Button>
              </CardActions>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <DataObjectIcon color="error" sx={{ mr: 1 }} />
                  <Typography variant="h5">
                    Expert
                  </Typography>
                </Box>
                <Typography variant="body2">
                  Specialized courses for security experts looking to master specific domains or cutting-edge security research and techniques.
                </Typography>
              </CardContent>
              <CardActions>
                <Button size="small" onClick={() => setSelectedCategory('expert')}>
                  View Expert Courses
                </Button>
              </CardActions>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};

export default CoursesPage; 