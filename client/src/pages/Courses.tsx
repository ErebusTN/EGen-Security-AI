import React from 'react';
import { CourseList } from '../components/training/CourseViewer';

const CoursesPage: React.FC = () => {
  return (
    <div className="courses-page">
      <CourseList />
    </div>
  );
};

export default CoursesPage; 